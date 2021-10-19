#!/usr/bin/env python

import subprocess
import os.path
import yaml
import io
import re

def parseKernelUsages(usageStr, usageDict):
    demangler = 'c++filt -p'

    def getKernelMem(usages):
        match = re.search(r"([0-9]+) bytes cmem\[0\]", usages)
        return match.group(1) if match else None
    def getSharedMem(usages):
        match = re.search(r"([0-9]+) bytes smem", usages)
        return match.group(1) if match else None
    def getRegisters(usages):
        match = re.search(r"[Uu]sed ([0-9]+) registers", usages)
        return match.group(1) if match else None
    def demangle(fn):
        expr = re.compile("__omp_offloading_[a-zA-Z0-9]*_[a-zA-Z0-9]*_(_Z.*_)_l[0-9]*$")
        match = expr.search(fn)
        function = match.group(1) if match else fn
        output = subprocess.run(demangler.split(' ') + [function], check=True, stdout=subprocess.PIPE)
        return output.stdout.decode('utf-8').strip()
    def getLine(fn):
        expr = re.compile("__omp_offloading_[a-zA-Z0-9]*_[a-zA-Z0-9]*_.*_l([0-9]*)$")
        match = expr.search(fn)
        return match.group(1) if match else 0

    expr = re.compile("Function properties for \'?([a-zA-Z0-9_]*)\'?\n(.*,.*)\n")
    for (fn, usages) in expr.findall(usageStr):
        info = usageDict[fn] if fn in usageDict else dict()
        info["Name"] = demangle(fn)
        info["DebugLoc"] = {"File" : "unknown", "Line": getLine(fn), "Column" : 0}
        info["Usage"] = {"Registers" : getRegisters(usages), "Shared" : getSharedMem(usages), "Kernel" : getKernelMem(usages)}
        usageDict[fn] = info

def getKernelUsage(stderr, fname='usage.yaml'):
    remarks = [line for line in stderr.split('\n') if re.search(r"^remark:", line)]
    ptxas = '\n'.join([line.split(':')[1].strip() for line in stderr.split('\n') if re.search(r"^ptxas info *:", line)])
    nvlink = '\n'.join([line.split(':')[1].strip() for line in stderr.split('\n') if re.search(r"^nvlink info *:", line)])

    if os.path.exists(fname):
        with io.open(fname, 'r', encoding = 'utf-8') as f:
            usage = yaml.load(f, Loader=yaml.Loader)
    else:
        usage = dict()

    parseKernelUsages(ptxas, usage)
    parseKernelUsages(nvlink, usage)

    return usage
