; RUN: llc -filetype=asm -o - -mtriple=armv6-unknown-linux-gnu < %s | FileCheck %s

define i32 @foo() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK-LABEL: Lxray_sled_0:
; CHECK-NEXT:  b  #20
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-LABEL: Ltmp0:
  ret i32 0
; CHECK-LABEL: Lxray_sled_1:
; CHECK-NEXT:  b  #20
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-LABEL: Ltmp1:
; CHECK-NEXT:  bx	lr
}
