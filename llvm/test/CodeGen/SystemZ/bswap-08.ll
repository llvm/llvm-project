; Verify that truncating stores do not use STRV
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i64 @llvm.bswap.i64(i64)

define void @f1(ptr %x, ptr %y) {
; CHECK-LABEL: f1:
; CHECK-NOT: strv
; CHECK: br %r14
  %a = load i64, ptr %y, align 8
  %b = tail call i64 @llvm.bswap.i64(i64 %a)
  %conv = trunc i64 %b to i32
  store i32 %conv, ptr %x, align 4
  ret void
}

