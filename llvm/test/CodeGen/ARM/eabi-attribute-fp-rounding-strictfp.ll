; Check how strictfp maps to attribute ABI_FP_rounding. The backend reports
; dynamic rounding is allowed (rounding attribute 19, 1) when any function
; definition is strictfp, i.e. may change the rounding mode at run time. With
; no strictfp function the module keeps the default rounding and the attribute
; is not emitted.

; RUN: split-file %s %t
; RUN: llc -mtriple=armv7-linux-gnueabi -mcpu=cortex-a15 < %t/strict.ll | FileCheck %s --check-prefix=STRICT
; RUN: llc -mtriple=armv7-linux-gnueabi -mcpu=cortex-a15 < %t/default.ll | FileCheck %s --check-prefix=DEFAULT

; STRICT: .eabi_attribute 19, 1 @ Tag_ABI_FP_rounding
; DEFAULT-NOT: .eabi_attribute 19

;--- strict.ll
define i32 @f0() strictfp {
entry:
  ret i32 42
}

define i32 @f1() {
entry:
  ret i32 42
}

;--- default.ll
define i32 @f0() {
entry:
  ret i32 42
}
