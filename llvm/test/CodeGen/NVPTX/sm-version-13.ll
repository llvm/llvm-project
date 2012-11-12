; RUN: llc < %s -march=nvptx -mcpu=sm_13 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_13 | FileCheck %s


; CHECK: .target sm_13

