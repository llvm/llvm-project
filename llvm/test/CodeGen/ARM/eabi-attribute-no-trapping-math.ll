; Check how no-trapping-math maps to attribute ABI_FP_exceptions across
; multiple functions. The backend only reports no-trapping-math (exceptions
; Not_Allowed, i.e. 21, 0) when every function definition agrees; a single
; disagreeing function taints the module back to the default (21, 1).

; RUN: split-file %s %t
; RUN: llc -mtriple=armv7-linux-gnueabi -mcpu=cortex-a15 < %t/agree.ll | FileCheck %s --check-prefix=AGREE
; RUN: llc -mtriple=armv7-linux-gnueabi -mcpu=cortex-a15 < %t/taint.ll | FileCheck %s --check-prefix=TAINT

; AGREE: .eabi_attribute 21, 0 @ Tag_ABI_FP_exceptions
; TAINT: .eabi_attribute 21, 1 @ Tag_ABI_FP_exceptions

;--- agree.ll
define i32 @f0() "no-trapping-math"="true" {
entry:
  ret i32 42
}

define i32 @f1() "no-trapping-math"="true" {
entry:
  ret i32 42
}

;--- taint.ll
define i32 @f0() "no-trapping-math"="true" {
entry:
  ret i32 42
}

define i32 @f1() {
entry:
  ret i32 42
}
