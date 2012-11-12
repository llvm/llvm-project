; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s


; CHECK: .target sm_20

