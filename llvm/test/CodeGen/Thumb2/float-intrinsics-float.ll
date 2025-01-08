; RUN: llc < %s -mtriple=thumbv7-none-eabi   -mcpu=cortex-m3                    | FileCheck %s -check-prefix=CHECK -check-prefix=SOFT -check-prefix=NONE
; RUN: llc < %s -mtriple=thumbv7-none-eabihf -mcpu=cortex-m4                    | FileCheck %s -check-prefix=CHECK -check-prefix=HARD -check-prefix=SP -check-prefix=NO-VMLA
; RUN: llc < %s -mtriple=thumbv7-none-eabihf -mcpu=cortex-m33                   | FileCheck %s -check-prefix=CHECK -check-prefix=HARD -check-prefix=SP -check-prefix=NO-VMLA
; RUN: llc < %s -mtriple=thumbv7-none-eabihf -mcpu=cortex-m7                    | FileCheck %s -check-prefix=CHECK -check-prefix=HARD -check-prefix=VFP  -check-prefix=FP-ARMv8  -check-prefix=VMLA
; RUN: llc < %s -mtriple=thumbv7-none-eabihf -mcpu=cortex-m7 -mattr=-fp64 | FileCheck %s -check-prefix=CHECK -check-prefix=HARD -check-prefix=SP -check-prefix=FP-ARMv8 -check-prefix=VMLA
; RUN: llc < %s -mtriple=thumbv7-none-eabihf -mcpu=cortex-a7                    | FileCheck %s -check-prefix=CHECK -check-prefix=HARD -check-prefix=NEON-A7 -check-prefix=VFP4 -check-prefix=NO-VMLA
; RUN: llc < %s -mtriple=thumbv7-none-eabihf -mcpu=cortex-a57                   | FileCheck %s -check-prefix=CHECK -check-prefix=HARD -check-prefix=NEON-A57 -check-prefix=FP-ARMv8 -check-prefix=VMLA

declare float     @llvm.sqrt.f32(float %Val)
define float @sqrt_f(float %a) {
; CHECK-LABEL: sqrt_f:
; SOFT: bl sqrtf
; HARD: vsqrt.f32 s0, s0
  %1 = call float @llvm.sqrt.f32(float %a)
  ret float %1
}

declare float     @llvm.powi.f32.i32(float %Val, i32 %power)
define float @powi_f(float %a, i32 %b) {
; CHECK-LABEL: powi_f:
; SOFT: bl __powisf2
; HARD: b __powisf2
  %1 = call float @llvm.powi.f32.i32(float %a, i32 %b)
  ret float %1
}

declare float     @llvm.sin.f32(float %Val)
define float @sin_f(float %a) {
; CHECK-LABEL: sin_f:
; SOFT: bl sinf
; HARD: b sinf
  %1 = call float @llvm.sin.f32(float %a)
  ret float %1
}

declare float     @llvm.cos.f32(float %Val)
define float @cos_f(float %a) {
; CHECK-LABEL: cos_f:
; SOFT: bl cosf
; HARD: b cosf
  %1 = call float @llvm.cos.f32(float %a)
  ret float %1
}

declare float     @llvm.tan.f32(float %Val)
define float @tan_f(float %a) {
; CHECK-LABEL: tan_f:
; SOFT: bl tanf
; HARD: b tanf
  %1 = call float @llvm.tan.f32(float %a)
  ret float %1
}

declare float     @llvm.pow.f32(float %Val, float %power)
define float @pow_f(float %a, float %b) {
; CHECK-LABEL: pow_f:
; SOFT: bl powf
; HARD: b powf
  %1 = call float @llvm.pow.f32(float %a, float %b)
  ret float %1
}

declare float     @llvm.exp.f32(float %Val)
define float @exp_f(float %a) {
; CHECK-LABEL: exp_f:
; SOFT: bl expf
; HARD: b expf
  %1 = call float @llvm.exp.f32(float %a)
  ret float %1
}

declare float     @llvm.exp2.f32(float %Val)
define float @exp2_f(float %a) {
; CHECK-LABEL: exp2_f:
; SOFT: bl exp2f
; HARD: b exp2f
  %1 = call float @llvm.exp2.f32(float %a)
  ret float %1
}

declare float     @llvm.log.f32(float %Val)
define float @log_f(float %a) {
; CHECK-LABEL: log_f:
; SOFT: bl logf
; HARD: b logf
  %1 = call float @llvm.log.f32(float %a)
  ret float %1
}

declare float     @llvm.log10.f32(float %Val)
define float @log10_f(float %a) {
; CHECK-LABEL: log10_f:
; SOFT: bl log10f
; HARD: b log10f
  %1 = call float @llvm.log10.f32(float %a)
  ret float %1
}

declare float     @llvm.log2.f32(float %Val)
define float @log2_f(float %a) {
; CHECK-LABEL: log2_f:
; SOFT: bl log2f
; HARD: b log2f
  %1 = call float @llvm.log2.f32(float %a)
  ret float %1
}

declare float     @llvm.fma.f32(float %a, float %b, float %c)
define float @fma_f(float %a, float %b, float %c) {
; CHECK-LABEL: fma_f:
; SOFT: bl fmaf
; HARD: vfma.f32
  %1 = call float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %1
}

declare float     @llvm.fabs.f32(float %Val)
define float @abs_f(float %a) {
; CHECK-LABEL: abs_f:
; SOFT: bic r0, r0, #-2147483648
; HARD: vabs.f32
  %1 = call float @llvm.fabs.f32(float %a)
  ret float %1
}

declare float     @llvm.copysign.f32(float  %Mag, float  %Sgn)
define float @copysign_f(float %a, float %b) {
; CHECK-LABEL: copysign_f:
; NONE: lsrs [[REG:r[0-9]+]], r{{[0-9]+}}, #31
; NONE: bfi r{{[0-9]+}}, [[REG]], #31, #1
; SP: lsrs [[REG:r[0-9]+]], r{{[0-9]+}}, #31
; SP: bfi r{{[0-9]+}}, [[REG]], #31, #1
; VFP: lsrs [[REG:r[0-9]+]], r{{[0-9]+}}, #31
; VFP: bfi r{{[0-9]+}}, [[REG]], #31, #1
; NEON-A7:       @ %bb.0:
; NEON-A7-NEXT:    vmov.f32 s2, s1
; NEON-A7-NEXT:    @ kill: def $s0 killed $s0 def $d0
; NEON-A7-NEXT:    vmov.i32 d16, #0x80000000
; NEON-A7-NEXT:    vbit d0, d1, d16
; NEON-A7-NEXT:    @ kill: def $s0 killed $s0 killed $d0
; NEON-A7-NEXT:    bx lr
; NEON-A57:       @ %bb.0:
; NEON-A57-NEXT:    vmov.f32 s2, s1
; NEON-A57-NEXT:    vmov.i32 d16, #0x80000000
; NEON-A57-NEXT:    @ kill: def $s0 killed $s0 def $d0
; NEON-A57-NEXT:    vbit d0, d1, d16
; NEON-A57-NEXT:    @ kill: def $s0 killed $s0 killed $d0
; NEON-A57-NEXT:    bx lr
  %1 = call float @llvm.copysign.f32(float %a, float %b)
  ret float %1
}

declare float     @llvm.floor.f32(float %Val)
define float @floor_f(float %a) {
; CHECK-LABEL: floor_f:
; SOFT: bl floorf
; VFP4: b floorf
; FP-ARMv8: vrintm.f32
  %1 = call float @llvm.floor.f32(float %a)
  ret float %1
}

declare float     @llvm.ceil.f32(float %Val)
define float @ceil_f(float %a) {
; CHECK-LABEL: ceil_f:
; SOFT: bl ceilf
; VFP4: b ceilf
; FP-ARMv8: vrintp.f32
  %1 = call float @llvm.ceil.f32(float %a)
  ret float %1
}

declare float     @llvm.trunc.f32(float %Val)
define float @trunc_f(float %a) {
; CHECK-LABEL: trunc_f:
; SOFT: bl truncf
; VFP4: b truncf
; FP-ARMv8: vrintz.f32
  %1 = call float @llvm.trunc.f32(float %a)
  ret float %1
}

declare float     @llvm.rint.f32(float %Val)
define float @rint_f(float %a) {
; CHECK-LABEL: rint_f:
; SOFT: bl rintf
; VFP4: b rintf
; FP-ARMv8: vrintx.f32
  %1 = call float @llvm.rint.f32(float %a)
  ret float %1
}

declare float     @llvm.nearbyint.f32(float %Val)
define float @nearbyint_f(float %a) {
; CHECK-LABEL: nearbyint_f:
; SOFT: bl nearbyintf
; VFP4: b nearbyintf
; FP-ARMv8: vrintr.f32
  %1 = call float @llvm.nearbyint.f32(float %a)
  ret float %1
}

declare float     @llvm.round.f32(float %Val)
define float @round_f(float %a) {
; CHECK-LABEL: round_f:
; SOFT: bl roundf
; VFP4: b roundf
; FP-ARMv8: vrinta.f32
  %1 = call float @llvm.round.f32(float %a)
  ret float %1
}

declare float     @llvm.fmuladd.f32(float %a, float %b, float %c)
define float @fmuladd_f(float %a, float %b, float %c) {
; CHECK-LABEL: fmuladd_f:
; SOFT: bl __aeabi_fmul
; SOFT: bl __aeabi_fadd
; VMLA: vfma.f32
; NO-VMLA: vmul.f32
; NO-VMLA: vadd.f32
  %1 = call float @llvm.fmuladd.f32(float %a, float %b, float %c)
  ret float %1
}

declare i16 @llvm.convert.to.fp16.f32(float %a)
define i16 @f_to_h(float %a) {
; CHECK-LABEL: f_to_h:
; SOFT: bl __aeabi_f2h
; HARD: vcvt{{[bt]}}.f16.f32
  %1 = call i16 @llvm.convert.to.fp16.f32(float %a)
  ret i16 %1
}

declare float @llvm.convert.from.fp16.f32(i16 %a)
define float @h_to_f(i16 %a) {
; CHECK-LABEL: h_to_f:
; SOFT: bl __aeabi_h2f
; HARD: vcvt{{[bt]}}.f32.f16
  %1 = call float @llvm.convert.from.fp16.f32(i16 %a)
  ret float %1
}
