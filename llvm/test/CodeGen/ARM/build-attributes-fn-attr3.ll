; Check FP options -fno-trapping-math and -fdenormal-fp-math. They are passed
; as function attributes, which map on to build attributes ABI_FP_exceptions
; ABI_FP_denormal. In the backend we therefore have a check to see if all
; functions have consistent function attributes values.
; Here we check values no-trapping-math=true and denormal-fp-math=ieee.

; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a15  | FileCheck %s

; CHECK: .eabi_attribute 20, 1
; CHECK: .eabi_attribute 21, 0

define i32 @foo() local_unnamed_addr #0 {
entry:
  ret i32 42
}

attributes #0 = { minsize norecurse nounwind optsize readnone "no-trapping-math"="true" "denormal-fp-math"="ieee"}
