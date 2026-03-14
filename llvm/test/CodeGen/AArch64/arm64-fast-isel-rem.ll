; RUN: llc -O0 -fast-isel -fast-isel-abort=1 -verify-machineinstrs -mtriple=arm64-apple-darwin < %s | FileCheck %s
; RUN: llc %s -O0 -fast-isel -fast-isel-abort=1 -mtriple=arm64-apple-darwin -print-after=finalize-isel -o /dev/null 2> %t
; RUN: FileCheck %s < %t --check-prefix=CHECK-SSA

; CHECK-SSA-LABEL: Machine code for function t1

; CHECK-SSA: [[QUOTREG:%[0-9]+]]:gpr32 = SDIVWr
; CHECK-SSA-NOT: [[QUOTREG]] =
; CHECK-SSA: {{%[0-9]+}}:gpr32 = MSUBWrrr [[QUOTREG]]

; CHECK-SSA-LABEL: Machine code for function t2

define i32 @t1(i32 %a, i32 %b) {
; CHECK: @t1
; CHECK: sdiv [[TMP:w[0-9]+]], w0, w1
; CHECK: msub w0, [[TMP]], w1, w0
  %1 = srem i32 %a, %b
  ret i32 %1
}

define i64 @t2(i64 %a, i64 %b) {
; CHECK: @t2
; CHECK: sdiv [[TMP:x[0-9]+]], x0, x1
; CHECK: msub x0, [[TMP]], x1, x0
  %1 = srem i64 %a, %b
  ret i64 %1
}

define i32 @t3(i32 %a, i32 %b) {
; CHECK: @t3
; CHECK: udiv [[TMP:w[0-9]+]], w0, w1
; CHECK: msub w0, [[TMP]], w1, w0
  %1 = urem i32 %a, %b
  ret i32 %1
}

define i64 @t4(i64 %a, i64 %b) {
; CHECK: @t4
; CHECK: udiv [[TMP:x[0-9]+]], x0, x1
; CHECK: msub x0, [[TMP]], x1, x0
  %1 = urem i64 %a, %b
  ret i64 %1
}
