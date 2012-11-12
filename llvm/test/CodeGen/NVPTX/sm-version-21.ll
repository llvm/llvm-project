; RUN: llc < %s -march=nvptx -mcpu=sm_21 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_21 | FileCheck %s


; CHECK: .target sm_21

