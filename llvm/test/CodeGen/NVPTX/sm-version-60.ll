; RUN: llc < %s -march=nvptx -mcpu=sm_60 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_60 | FileCheck %s

; CHECK: .version 5.0
; CHECK: .target sm_60
