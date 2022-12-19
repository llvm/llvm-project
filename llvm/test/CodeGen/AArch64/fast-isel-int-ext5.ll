; RUN: llc -mtriple=aarch64-apple-darwin -O0 -fast-isel -fast-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: int_ext_opt
define i64 @int_ext_opt(ptr %addr, i1 %c1, i1 %c2) {
entry:
  %0 = load i8, ptr %addr
  br i1 %c1, label %bb1, label %bb2

bb1:
  %1 = zext i8 %0 to i64
  br i1 %c2, label %bb2, label %exit

bb2:
  %2 = phi i64 [1, %entry], [%1, %bb1]
  ret i64 %2

exit:
  ret i64 0
}
