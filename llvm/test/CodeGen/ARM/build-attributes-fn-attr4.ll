; Check FP option -fdenormal-fp-math. This passed as function attribute,
; which map on to build attributes ABI_FP_denormal. In the backend we
; therefore have a check to see if all functions have consistent function
; attributes values.
; Here we check the denormal-fp-math=positive-zero value.

; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a15  | FileCheck %s

; CHECK: .eabi_attribute 20, 0

define i32 @foo1() local_unnamed_addr #0 {
entry:
  ret i32 42
}

attributes #0 = { minsize norecurse nounwind optsize readnone "denormal-fp-math"="positive-zero"}
