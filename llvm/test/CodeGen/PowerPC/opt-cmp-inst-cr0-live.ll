; RUN: llc -verify-machineinstrs -print-before=peephole-opt -print-after=peephole-opt -mtriple=powerpc64-unknown-linux-gnu -o /dev/null 2>&1 < %s | FileCheck %s

define signext i32 @fn1(i32 %baz) {
  %1 = mul nsw i32 %baz, 208
  %2 = zext i32 %1 to i64
  %3 = shl i64 %2, 48
  %4 = ashr exact i64 %3, 48
; CHECK: ANDIo8 {{[^,]+}}, 65520, %CR0<imp-def,dead>;
; CHECK: CMPLDI
; CHECK: BCC

; CHECK: ANDIo8 {{[^,]+}}, 65520, %CR0<imp-def>;
; CHECK: COPY %CR0
; CHECK: BCC
  %5 = icmp eq i64 %4, 0
  br i1 %5, label %foo, label %bar

foo:
  ret i32 1

bar:
  ret i32 0
}
