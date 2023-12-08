; RUN: llc -mcpu=generic -O1 -relocation-model=pic < %s | FileCheck %s
target datalayout = "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128"
target triple = "i686-apple-darwin"

define i64 @test_lshr() {
entry:
; CHECK-NOT: movl $-1, 16(%esp)
; CHECK-NOT: movl  $-1, %eax
  %retval = alloca i64
  %op1 = alloca i64
  %op2 = alloca i64
  store i64 -6687208052682386272, ptr %op1
  store i64 7106745059734980448, ptr %op2
  %tmp1 = load i64, ptr %op1
  %tmp2 = load i64, ptr %op2
  %tmp = xor i64 %tmp2, 7106745059734980448
  %tmp3 = lshr i64 %tmp1, %tmp
  store i64 %tmp3, ptr %retval
  %tmp4 = load i64, ptr %retval
  ret i64 %tmp4
}
