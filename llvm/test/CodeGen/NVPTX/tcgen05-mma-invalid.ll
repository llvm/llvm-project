; RUN: not llc < %s -mtriple=nvptx64 -mcpu=sm_100a -o /dev/null 2>&1 | FileCheck %s
; XFAIL: *
target triple = "nvptx64-nvidia-cuda"

