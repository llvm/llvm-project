; RUN: llvm-link -S -internalize %s %p/Inputs/internalize-lazy.ll | FileCheck %s

; CHECK: define internal void @f
; CHECK: define internal void @g
