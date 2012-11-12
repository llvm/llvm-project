; RUN: llc < %s -march=nvptx -mcpu=sm_12 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_12 | FileCheck %s


; CHECK: .target sm_12

