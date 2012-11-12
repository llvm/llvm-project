; RUN: llc < %s -march=nvptx -mcpu=sm_20 -mattr=ptx30 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 -mattr=ptx30 | FileCheck %s


; CHECK: .version 3.0

