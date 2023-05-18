; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2,+v7 %s -o - | FileCheck %s

define i32 @f1(i32 %a) {
; CHECK-LABEL: f1:
; CHECK: clz r
    %tmp = tail call i32 @llvm.ctlz.i32(i32 %a, i1 true)
    ret i32 %tmp
}

declare i32 @llvm.ctlz.i32(i32, i1) nounwind readnone
