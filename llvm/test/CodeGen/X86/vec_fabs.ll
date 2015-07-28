; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=avx | FileCheck %s


define <2 x double> @fabs_v2f64(<2 x double> %p)
{
  ; CHECK-LABEL: fabs_v2f64
  ; CHECK: vandpd
  %t = call <2 x double> @llvm.fabs.v2f64(<2 x double> %p)
  ret <2 x double> %t
}
declare <2 x double> @llvm.fabs.v2f64(<2 x double> %p)

define <4 x float> @fabs_v4f32(<4 x float> %p)
{
  ; CHECK-LABEL: fabs_v4f32
  ; CHECK: vandps
  %t = call <4 x float> @llvm.fabs.v4f32(<4 x float> %p)
  ret <4 x float> %t
}
declare <4 x float> @llvm.fabs.v4f32(<4 x float> %p)

define <4 x double> @fabs_v4f64(<4 x double> %p)
{
  ; CHECK-LABEL: fabs_v4f64
  ; CHECK: vandpd
  %t = call <4 x double> @llvm.fabs.v4f64(<4 x double> %p)
  ret <4 x double> %t
}
declare <4 x double> @llvm.fabs.v4f64(<4 x double> %p)

define <8 x float> @fabs_v8f32(<8 x float> %p)
{
  ; CHECK-LABEL: fabs_v8f32
  ; CHECK: vandps
  %t = call <8 x float> @llvm.fabs.v8f32(<8 x float> %p)
  ret <8 x float> %t
}
declare <8 x float> @llvm.fabs.v8f32(<8 x float> %p)

; PR20354: when generating code for a vector fabs op,
; make sure that we're only turning off the sign bit of each float value.
; No constant pool loads or vector ops are needed for the fabs of a
; bitcasted integer constant; we should just return an integer constant
; that has the sign bits turned off.
;
; So instead of something like this:
;    movabsq (constant pool load of mask for sign bits) 
;    vmovq   (move from integer register to vector/fp register)
;    vandps  (mask off sign bits)
;    vmovq   (move vector/fp register back to integer return register)
;
; We should generate:
;    mov     (put constant value in return register)

define i64 @fabs_v2f32_1() {
; CHECK-LABEL: fabs_v2f32_1:
; CHECK: movabsq $9223372032559808512, %rax # imm = 0x7FFFFFFF00000000
; CHECK-NEXT: retq
 %bitcast = bitcast i64 18446744069414584320 to <2 x float> ; 0xFFFF_FFFF_0000_0000
 %fabs = call <2 x float> @llvm.fabs.v2f32(<2 x float> %bitcast)
 %ret = bitcast <2 x float> %fabs to i64
 ret i64 %ret
}

define i64 @fabs_v2f32_2() {
; CHECK-LABEL: fabs_v2f32_2:
; CHECK: movl $2147483647, %eax       # imm = 0x7FFFFFFF
; CHECK-NEXT: retq
 %bitcast = bitcast i64 4294967295 to <2 x float> ; 0x0000_0000_FFFF_FFFF
 %fabs = call <2 x float> @llvm.fabs.v2f32(<2 x float> %bitcast)
 %ret = bitcast <2 x float> %fabs to i64
 ret i64 %ret
}

declare <2 x float> @llvm.fabs.v2f32(<2 x float> %p)
