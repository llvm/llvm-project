; RUN: llc -mtriple=arm-eabi -mattr=+v7 %s -o - | FileCheck %s

define i32 @f(i32 %a) nounwind readnone optsize ssp {
entry:
  %conv = zext i32 %a to i64
  %tmp1 = tail call i64 @llvm.ctlz.i64(i64 %conv, i1 true)
; CHECK: clz
; CHECK-NOT: adds
  %cast = trunc i64 %tmp1 to i32
  %sub = sub nsw i32 63, %cast
  ret i32 %sub
}

declare i64 @llvm.ctlz.i64(i64, i1) nounwind readnone
