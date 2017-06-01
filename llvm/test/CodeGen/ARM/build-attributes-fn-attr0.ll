; Check FP options -fno-trapping-math and -fdenormal-fp-math. They are passed
; as function attributes, which map on to build attributes ABI_FP_exceptions
; ABI_FP_denormal. In the backend we therefore have a check to see if all
; functions have consistent function attributes values. This check also returns
; true when the compilation unit does not have any functions (i.e. the
; attributes are consistent), which is what we check with this regression test.

; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a15  | FileCheck %s

; CHECK:  .eabi_attribute 20, 2
; CHECK:  .eabi_attribute 21, 0
