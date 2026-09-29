; Check FP options -fno-trapping-math and -fdenormal-fp-math. They are passed
; as function attributes, which map on to build attributes ABI_FP_exceptions
; ABI_FP_denormal. In the backend we therefore have a check to see if all
; functions have consistent function attributes values.
; Here we test correct output for no-trapping-math=false

; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a15  | FileCheck %s

; CHECK: .eabi_attribute 21, 1

define i32 @foo() local_unnamed_addr #0 {
entry:
  ret i32 42
}

attributes #0 = { minsize norecurse nounwind optsize readnone "no-trapping-math"="false" }
