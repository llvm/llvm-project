; RUN: llc < %s -mtriple=armv7-eabi -float-abi=hard -mcpu=cortex-a8 | FileCheck %s

; CHECK-LABEL: test:
; CHECK:         vabs.f32        q0, q0
define <4 x float> @test(<4 x float> %a) {
  %foo = call <4 x float> @llvm.fabs.v4f32(<4 x float> %a)
  ret <4 x float> %foo
}
declare <4 x float> @llvm.fabs.v4f32(<4 x float> %a)

; CHECK-LABEL: test2:
; CHECK:        vabs.f32        d0, d0
define <2 x float> @test2(<2 x float> %a) {
  %foo = call <2 x float> @llvm.fabs.v2f32(<2 x float> %a)
    ret <2 x float> %foo
}
declare <2 x float> @llvm.fabs.v2f32(<2 x float> %a)

; No constant pool loads or vector ops are needed for the fabs of a
; bitcasted integer constant; we should just return integer constants
; that have the sign bits turned off.
;
; So instead of something like this:
; 	mvn	r0, #0
; 	mov	r1, #0
; 	vmov	d16, r1, r0
; 	vabs.f32	d16, d16
; 	vmov	r0, r1, d16
; 	bx	lr
;
; We should generate:
;	mov	r0, #0
;	mvn	r1, #-2147483648
;	bx	lr

define i64 @fabs_v2f32_1() {
; CHECK-LABEL: fabs_v2f32_1:
; CHECK: mvn r1, #-2147483648
; CHECK: bx lr
; CHECK-NOT: vabs
 %bitcast = bitcast i64 18446744069414584320 to <2 x float> ; 0xFFFF_FFFF_0000_0000
 %fabs = call <2 x float> @llvm.fabs.v2f32(<2 x float> %bitcast)
 %ret = bitcast <2 x float> %fabs to i64
 ret i64 %ret
}

define i64 @fabs_v2f32_2() {
; CHECK-LABEL: fabs_v2f32_2:
; CHECK: mvn r0, #-2147483648
; CHECK: bx lr
; CHECK-NOT: vabs
 %bitcast = bitcast i64 4294967295 to <2 x float> ; 0x0000_0000_FFFF_FFFF
 %fabs = call <2 x float> @llvm.fabs.v2f32(<2 x float> %bitcast)
 %ret = bitcast <2 x float> %fabs to i64
 ret i64 %ret
}
