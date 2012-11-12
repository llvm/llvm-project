; RUN: llc < %s -march=nvptx -mcpu=sm_11 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_11 | FileCheck %s


; CHECK: .target sm_11

