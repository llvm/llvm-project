#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import argparse
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=(1 << 32))
parser.add_argument('--optcmd', default=("opt"))
parser.add_argument('--filecheckcmd', default=("FileCheck"))
parser.add_argument('--prefix', default=("CHECK-BISECT"))
parser.add_argument('--test', default=(""))

args = parser.parse_args()

start = args.start
end = args.end

opt_command = [args.optcmd, "-O2", "-opt-bisect-limit=%(count)s", "-S", args.test]
check_command = [args.filecheckcmd, args.test, "--check-prefix=%s" % args.prefix]
last = None
while start != end and start != end-1:
    count = int(round(start + (end - start)/2))
    cmd = [x % {'count':count} for x in opt_command]
    print("opt: " + str(cmd))
    opt_result = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    filecheck_result = subprocess.Popen(check_command, stdin=opt_result.stdout)
    opt_result.stdout.close()
    opt_result.stderr.close()
    filecheck_result.wait()
    if filecheck_result.returncode == 0:
        start = count
    else:
        end = count

print("Last good count: %d" % start)
