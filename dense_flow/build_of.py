__author__ = 'yjxiong'

import glob
import cv2
import os
import sys
from multiprocessing import Pool, current_process
import argparse

out_path = ''
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
print 'CURRENT_DIR = {}'.format(CURRENT_DIR)

def dump_frames(vid_path):
    video = cv2.VideoCapture(vid_path)
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)

    fcount = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass
    file_list = []
    for i in xrange(fcount):
        ret, frame = video.read()
        assert ret
        cv2.imwrite('{}/{:06d}.jpg'.format(out_full_path, i), frame)
        access_path = '{}/{:06d}.jpg'.format(vid_name, i)
        file_list.append(access_path)
    print '{} done'.format(vid_name)
    return file_list


def run_optical_flow(vid_item, dev_id=0):
    vid_path = vid_item[0]
    vid_id = vid_item[1]
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass

    current = current_process()
    print ">>> current = {}".format(current)
    dev_id = int(current._identity[0]) - 1
    print ">>> dev_id = {}".format(dev_id)
    image_path = '{}/img'.format(out_full_path)
    flow_x_path = '{}/flow_x'.format(out_full_path)
    flow_y_path = '{}/flow_y'.format(out_full_path)

    cmd = '{}/build/extract_gpu -f={} -x={} -y={} -i={} -b=20 -t=1 -d={} -s=1 -o {} -w {} -h {}'.format(
        CURRENT_DIR, vid_path, flow_x_path, flow_y_path, image_path, dev_id, out_format, new_size[0],new_size[1])

    print ">>> cmd: {}".format(cmd)

    os.system(cmd)
    print '{} {} done'.format(vid_id, vid_name)
    sys.stdout.flush()
    return True

def run_warp_optical_flow(vid_item, dev_id=0):
    vid_path = vid_item[0]
    vid_id = vid_item[1]
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass

    current = current_process()
    dev_id = int(current._identity[0]) - 1
    flow_x_path = '{}/flow_x'.format(out_full_path)
    flow_y_path = '{}/flow_y'.format(out_full_path)
    cmd = '{}/build/extract_warp_gpu -f {} -x {} -y {} -b 20 -t 1 -d {} -s 1 -o {}'.format(
        CURRENT_DIR, vid_path, flow_x_path, flow_y_path, dev_id, out_format)
    print ">>> cmd: {}".format(cmd)
    os.system(cmd)
    print 'warp on {} {} done'.format(vid_id, vid_name)
    return True

## For CPU user:
def run_optical_flow_CPU(vid_item, dev_id=0):
    vid_path = vid_item[0]
    vid_id = vid_item[1]
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass
    current = current_process()
    print ">>> current = {}".format(current)
    dev_id = int(current._identity[0]) - 1
    print ">>> dev_id = {}".format(dev_id)
    image_path = '{}/img'.format(out_full_path)
    flow_x_path = '{}/flow_x'.format(out_full_path)
    flow_y_path = '{}/flow_y'.format(out_full_path)
    try: 
        os.mkdir(image_path)
        os.mkdir(flow_x_path)
        os.mkdir(flow_y_path)
    except OSError:
        pass
    cmd = '{}/build/extract_cpu -f={} -x={} -y={} -i={} -b=20 -t=1 -d={} -s=1 -o {} -w {} -h {}'.format(
        CURRENT_DIR, vid_path, flow_x_path, flow_y_path, image_path, dev_id, out_format, new_size[0],new_size[1])
    print '>>> cmd: {}'.format(cmd)
    os.system(cmd)
    print '{} {} done'.format(vid_id, vid_name)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="extract optical flows. ie, python {}/{} --src_dir /data/UCF-101 --out_dir /data/ucf101_frames --flow_type tvl1".format(
            CURRENT_DIR, __file__))
    parser.add_argument("--src_dir", type=str, default='./UCF-101',
                        help='path to the video data')
    parser.add_argument("--out_dir", type=str, default='./ucf101_frames',
                        help='path to store frames and optical flow')
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--flow_type", type=str, default='tvl1', 
                        choices=['tvl1', 'warp_tvl1', 'cpu_tvl1'])
    parser.add_argument("--new_width", type=int, default=0, help='resize image width')
    parser.add_argument("--new_height", type=int, default=0, help='resize image height')
    parser.add_argument("--out_format", type=str, default='dir', choices=['dir','zip'])

    args = parser.parse_args()

    out_path = args.out_dir
    src_path = args.src_dir
    num_worker = args.num_worker
    flow_type = args.flow_type
    new_size = (args.new_width, args.new_height)
    out_format = args.out_format


    vid_list = glob.glob(src_path+'/*/*.mp4')
    vid_list.extend(glob.glob(src_path+'/*/*.avi'))
    print "{} of video detected".format(len(vid_list))
    pool = Pool(num_worker)
    if flow_type == 'tvl1':
        pool.map(run_optical_flow, zip(vid_list, xrange(len(vid_list))))
    elif flow_type == 'warp_tvl1':
        pool.map(run_warp_optical_flow, zip(vid_list, xrange(len(vid_list))))
    ## CPU:
    elif flow_type == 'cpu_tvl1':
        pool.map(run_optical_flow_CPU, zip(vid_list, xrange(len(vid_list))))
    