; RUN: llc -mtriple=armv8a-none-eabi %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-SP,CHECK-DP
; RUN: llc -mtriple=thumbv8m.main-none-eabi %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NOSP,CHECK-NODP
; RUN: llc -mtriple=thumbv8m.main-none-eabi %s -o - -mattr=fp-armv8 | FileCheck %s --check-prefixes=CHECK,CHECK-SP,CHECK-DP
; RUN: llc -mtriple=thumbv8m.main-none-eabi %s -o - -mattr=fp-armv8sp | FileCheck %s --check-prefixes=CHECK,CHECK-SP,CHECK-NODP

; Check that constrained fp intrinsics are correctly lowered. In particular
; check that the valid combinations of single-precision and double-precision
; hardware being present or absent work as expected (i.e. we get an instruction
; when one is available, otherwise a libcall).

; FIXME: Tests fails as various things in CodeGen and Target/ARM need fixing.
; XFAIL: *


; Single-precision intrinsics

; CHECK-LABEL: add_f32:
; CHECK-NOSP: bl __aeabi_fadd
; CHECK-SP: vadd.f32
define float @add_f32(float %x, float %y) #0 {
  %val = call float @llvm.experimental.constrained.fadd.f32(float %x, float %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: sub_f32:
; CHECK-NOSP: bl __aeabi_fsub
; CHECK-SP: vsub.f32
define float @sub_f32(float %x, float %y) #0 {
  %val = call float @llvm.experimental.constrained.fsub.f32(float %x, float %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: mul_f32:
; CHECK-NOSP: bl __aeabi_fmul
; CHECK-SP: vmul.f32
define float @mul_f32(float %x, float %y) #0 {
  %val = call float @llvm.experimental.constrained.fmul.f32(float %x, float %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: div_f32:
; CHECK-NOSP: bl __aeabi_fdiv
; CHECK-SP: vdiv.f32
define float @div_f32(float %x, float %y) #0 {
  %val = call float @llvm.experimental.constrained.fdiv.f32(float %x, float %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: frem_f32:
; CHECK: bl fmodf
define float @frem_f32(float %x, float %y) #0 {
  %val = call float @llvm.experimental.constrained.frem.f32(float %x, float %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: fma_f32:
; CHECK-NOSP: bl fmaf
; CHECK-SP: vfma.f32
define float @fma_f32(float %x, float %y, float %z) #0 {
  %val = call float @llvm.experimental.constrained.fma.f32(float %x, float %y, float %z, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: fptosi_f32:
; CHECK-NOSP: bl __aeabi_f2iz
; CHECK-SP: vcvt.s32.f32
define i32 @fptosi_f32(float %x) #0 {
  %val = call i32 @llvm.experimental.constrained.fptosi.f32(float %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: fptoui_f32:
; CHECK-NOSP: bl __aeabi_f2uiz
; CHECK-SP: vcvt.u32.f32
define i32 @fptoui_f32(float %x) #0 {
  %val = call i32 @llvm.experimental.constrained.fptoui.f32(float %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: sqrt_f32:
; CHECK-NOSP: bl sqrtf
; CHECK-SP: vsqrt.f32
define float @sqrt_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.sqrt.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: powi_f32:
; CHECK: bl __powisf2
define float @powi_f32(float %x, i32 %y) #0 {
  %val = call float @llvm.experimental.constrained.powi.f32(float %x, i32 %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: sin_f32:
; CHECK: bl sinf
define float @sin_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.sin.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: cos_f32:
; CHECK: bl cosf
define float @cos_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.cos.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: pow_f32:
; CHECK: bl powf
define float @pow_f32(float %x, float %y) #0 {
  %val = call float @llvm.experimental.constrained.pow.f32(float %x, float %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: log_f32:
; CHECK: bl logf
define float @log_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.log.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: log10_f32:
; CHECK: bl log10f
define float @log10_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.log10.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: log2_f32:
; CHECK: bl log2f
define float @log2_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.log2.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: exp_f32:
; CHECK: bl expf
define float @exp_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.exp.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: exp2_f32:
; CHECK: bl exp2f
define float @exp2_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.exp2.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: rint_f32:
; CHECK-NOSP: bl rintf
; CHECK-SP: vrintx.f32
define float @rint_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.rint.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: nearbyint_f32:
; CHECK-NOSP: bl nearbyintf
; CHECK-SP: vrintr.f32
define float @nearbyint_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.nearbyint.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: lrint_f32:
; CHECK: bl lrintf
define i32 @lrint_f32(float %x) #0 {
  %val = call i32 @llvm.experimental.constrained.lrint.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: llrint_f32:
; CHECK: bl llrintf
define i32 @llrint_f32(float %x) #0 {
  %val = call i32 @llvm.experimental.constrained.llrint.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: maxnum_f32:
; CHECK-NOSP: bl fmaxf
; CHECK-SP: vmaxnm.f32
define float @maxnum_f32(float %x, float %y) #0 {
  %val = call float @llvm.experimental.constrained.maxnum.f32(float %x, float %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: minnum_f32:
; CHECK-NOSP: bl fminf
; CHECK-SP: vminnm.f32
define float @minnum_f32(float %x, float %y) #0 {
  %val = call float @llvm.experimental.constrained.minnum.f32(float %x, float %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: ceil_f32:
; CHECK-NOSP: bl ceilf
; CHECK-SP: vrintp.f32
define float @ceil_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.ceil.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: floor_f32:
; CHECK-NOSP: bl floorf
; CHECK-SP: vrintm.f32
define float @floor_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.floor.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: lround_f32:
; CHECK: bl lroundf
define i32 @lround_f32(float %x) #0 {
  %val = call i32 @llvm.experimental.constrained.lround.f32(float %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: llround_f32:
; CHECK: bl llroundf
define i32 @llround_f32(float %x) #0 {
  %val = call i32 @llvm.experimental.constrained.llround.f32(float %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: round_f32:
; CHECK-NOSP: bl roundf
; CHECK-SP: vrinta.f32
define float @round_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.round.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: trunc_f32:
; CHECK-NOSP: bl truncf
; CHECK-SP: vrintz.f32
define float @trunc_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.trunc.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}


; Double-precision intrinsics

; CHECK-LABEL: add_f64:
; CHECK-NODP: bl __aeabi_dadd
; CHECK-DP: vadd.f64
define double @add_f64(double %x, double %y) #0 {
  %val = call double @llvm.experimental.constrained.fadd.f64(double %x, double %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: sub_f64:
; CHECK-NODP: bl __aeabi_dsub
; CHECK-DP: vsub.f64
define double @sub_f64(double %x, double %y) #0 {
  %val = call double @llvm.experimental.constrained.fsub.f64(double %x, double %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: mul_f64:
; CHECK-NODP: bl __aeabi_dmul
; CHECK-DP: vmul.f64
define double @mul_f64(double %x, double %y) #0 {
  %val = call double @llvm.experimental.constrained.fmul.f64(double %x, double %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: div_f64:
; CHECK-NODP: bl __aeabi_ddiv
; CHECK-DP: vdiv.f64
define double @div_f64(double %x, double %y) #0 {
  %val = call double @llvm.experimental.constrained.fdiv.f64(double %x, double %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: frem_f64:
; CHECK: bl fmod
define double @frem_f64(double %x, double %y) #0 {
  %val = call double @llvm.experimental.constrained.frem.f64(double %x, double %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: fma_f64:
; CHECK-NODP: bl fma
; CHECK-DP: vfma.f64
define double @fma_f64(double %x, double %y, double %z) #0 {
  %val = call double @llvm.experimental.constrained.fma.f64(double %x, double %y, double %z, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: fptosi_f64:
; CHECK-NODP: bl __aeabi_d2iz
; CHECK-DP: vcvt.s32.f64
define i32 @fptosi_f64(double %x) #0 {
  %val = call i32 @llvm.experimental.constrained.fptosi.f64(double %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: fptoui_f64:
; CHECK-NODP: bl __aeabi_d2uiz
; CHECK-DP: vcvt.u32.f64
define i32 @fptoui_f64(double %x) #0 {
  %val = call i32 @llvm.experimental.constrained.fptoui.f64(double %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: sqrt_f64:
; CHECK-NODP: bl sqrt
; CHECK-DP: vsqrt.f64
define double @sqrt_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.sqrt.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: powi_f64:
; CHECK: bl __powidf2
define double @powi_f64(double %x, i32 %y) #0 {
  %val = call double @llvm.experimental.constrained.powi.f64(double %x, i32 %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: sin_f64:
; CHECK: bl sin
define double @sin_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.sin.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: cos_f64:
; CHECK: bl cos
define double @cos_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.cos.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: pow_f64:
; CHECK: bl pow
define double @pow_f64(double %x, double %y) #0 {
  %val = call double @llvm.experimental.constrained.pow.f64(double %x, double %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: log_f64:
; CHECK: bl log
define double @log_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.log.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: log10_f64:
; CHECK: bl log10
define double @log10_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.log10.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: log2_f64:
; CHECK: bl log2
define double @log2_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.log2.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: exp_f64:
; CHECK: bl exp
define double @exp_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.exp.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: exp2_f64:
; CHECK: bl exp2
define double @exp2_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.exp2.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: rint_f64:
; CHECK-NODP: bl rint
; CHECK-DP: vrintx.f64
define double @rint_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.rint.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: nearbyint_f64:
; CHECK-NODP: bl nearbyint
; CHECK-DP: vrintr.f64
define double @nearbyint_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.nearbyint.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: lrint_f64:
; CHECK: bl lrint
define i32 @lrint_f64(double %x) #0 {
  %val = call i32 @llvm.experimental.constrained.lrint.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: llrint_f64:
; CHECK: bl llrint
define i32 @llrint_f64(double %x) #0 {
  %val = call i32 @llvm.experimental.constrained.llrint.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: maxnum_f64:
; CHECK-NODP: bl fmax
; CHECK-DP: vmaxnm.f64
define double @maxnum_f64(double %x, double %y) #0 {
  %val = call double @llvm.experimental.constrained.maxnum.f64(double %x, double %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: minnum_f64:
; CHECK-NODP: bl fmin
; CHECK-DP: vminnm.f64
define double @minnum_f64(double %x, double %y) #0 {
  %val = call double @llvm.experimental.constrained.minnum.f64(double %x, double %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: ceil_f64:
; CHECK-NODP: bl ceil
; CHECK-DP: vrintp.f64
define double @ceil_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.ceil.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: floor_f64:
; CHECK-NODP: bl floor
; CHECK-DP: vrintm.f64
define double @floor_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.floor.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: lround_f64:
; CHECK: bl lround
define i32 @lround_f64(double %x) #0 {
  %val = call i32 @llvm.experimental.constrained.lround.f64(double %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: llround_f64:
; CHECK: bl llround
define i32 @llround_f64(double %x) #0 {
  %val = call i32 @llvm.experimental.constrained.llround.f64(double %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: round_f64:
; CHECK-NODP: bl round
; CHECK-DP: vrinta.f64
define double @round_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.round.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: trunc_f64:
; CHECK-NODP: bl trunc
; CHECK-DP: vrintz.f64
define double @trunc_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.trunc.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}


; Single/Double conversion intrinsics

; CHECK-LABEL: fptrunc_f32:
; CHECK-NODP: bl __aeabi_d2f
; CHECK-DP: vcvt.f32.f64
define float @fptrunc_f32(double %x) #0 {
  %val = call float @llvm.experimental.constrained.fptrunc.f32.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: fpext_f32:
; CHECK-NODP: bl __aeabi_f2d
; CHECK-DP: vcvt.f64.f32
define double @fpext_f32(float %x) #0 {
  %val = call double @llvm.experimental.constrained.fpext.f64.f32(float %x, metadata !"fpexcept.strict") #0
  ret double %val
}


attributes #0 = { strictfp }

declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fsub.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fmul.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fdiv.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.frem.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fma.f32(float, float, float, metadata, metadata)
declare i32 @llvm.experimental.constrained.fptosi.f32(float, metadata)
declare i32 @llvm.experimental.constrained.fptoui.f32(float, metadata)
declare float @llvm.experimental.constrained.sqrt.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.powi.f32(float, i32, metadata, metadata)
declare float @llvm.experimental.constrained.sin.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.cos.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.pow.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.log.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.log10.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.log2.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.exp.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.exp2.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.rint.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.nearbyint.f32(float, metadata, metadata)
declare i32 @llvm.experimental.constrained.lrint.f32(float, metadata, metadata)
declare i32 @llvm.experimental.constrained.llrint.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.maxnum.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.minnum.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.ceil.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.floor.f32(float, metadata, metadata)
declare i32 @llvm.experimental.constrained.lround.f32(float, metadata)
declare i32 @llvm.experimental.constrained.llround.f32(float, metadata)
declare float @llvm.experimental.constrained.round.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.trunc.f32(float, metadata, metadata)

declare double @llvm.experimental.constrained.fadd.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.fsub.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.fmul.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.fdiv.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.frem.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.fma.f64(double, double, double, metadata, metadata)
declare i32 @llvm.experimental.constrained.fptosi.f64(double, metadata)
declare i32 @llvm.experimental.constrained.fptoui.f64(double, metadata)
declare double @llvm.experimental.constrained.sqrt.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.powi.f64(double, i32, metadata, metadata)
declare double @llvm.experimental.constrained.sin.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.cos.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.pow.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.log.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.log10.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.log2.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.exp.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.exp2.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.rint.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.nearbyint.f64(double, metadata, metadata)
declare i32 @llvm.experimental.constrained.lrint.f64(double, metadata, metadata)
declare i32 @llvm.experimental.constrained.llrint.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.maxnum.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.minnum.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.ceil.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.floor.f64(double, metadata, metadata)
declare i32 @llvm.experimental.constrained.lround.f64(double, metadata)
declare i32 @llvm.experimental.constrained.llround.f64(double, metadata)
declare double @llvm.experimental.constrained.round.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.trunc.f64(double, metadata, metadata)

declare float @llvm.experimental.constrained.fptrunc.f32.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.fpext.f64.f32(float, metadata)
