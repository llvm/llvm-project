; RUN: llc -mtriple=armv7-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -mtriple=armv7-apple-ios6.0.0    < %s | FileCheck %s

define i32 @callee() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK:       .p2align	2
; CHECK-LABEL: Lxray_sled_0:
; CHECK-NEXT:  b	#20
; CHECK-COUNT-6: nop
; CHECK-NEXT:  Ltmp[[#]]:
  ret i32 0
; CHECK-NEXT:  mov	r0, #0
; CHECK-NEXT:  .p2align	2
; CHECK-LABEL: Lxray_sled_1:
; CHECK-NEXT:  b	#20
; CHECK-COUNT-6: nop
; CHECK-NEXT:  Ltmp[[#]]:
; CHECK-NEXT:  bx	lr
}

define i32 @caller() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK:       .p2align	2
; CHECK-LABEL: Lxray_sled_2:
; CHECK-NEXT:  b	#20
; CHECK-COUNT-6: nop
; CHECK-NEXT:  Ltmp[[#]]:
; CHECK:       .p2align	2
; CHECK-LABEL: Lxray_sled_3:
; CHECK-NEXT:  b	#20
; CHECK-COUNT-6: nop
; CHECK-NEXT:  Ltmp[[#]]:
  %retval = tail call i32 @callee()
; CHECK:       b	{{.*}}callee
  ret i32 %retval
}
