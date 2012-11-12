; RUN: llc < %s -march=thumb -mattr=+thumb2,+v7 | FileCheck %s

define i32 @f1(i32 %a) {
; CHECK: f1:
; CHECK: clz r
    %tmp = tail call i32 @llvm.ctlz.i32(i32 %a, i1 true)
    ret i32 %tmp
}

declare i32 @llvm.ctlz.i32(i32, i1) nounwind readnone
