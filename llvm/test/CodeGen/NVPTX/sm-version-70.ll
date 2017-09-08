; RUN: llc < %s -march=nvptx -mcpu=sm_70 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_70 | FileCheck %s

; CHECK: .version 6.0
; CHECK: .target sm_70
