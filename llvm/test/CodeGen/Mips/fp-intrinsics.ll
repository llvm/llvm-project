; RUN: llc -mtriple=mips -mcpu=mips32r2 < %s | FileCheck %s -check-prefixes=CHECK,CHECK-R2
; RUN: llc -mtriple=mips -mcpu=mips32r6 < %s | FileCheck %s -check-prefixes=CHECK,CHECK-R6
; RUN: llc -mtriple=mips -mcpu=mips32r2 -mattr=+fp64,+fpxx -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-R2
; RUN: llc -mtriple=mips -mcpu=mips64r2 -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-R2

; Single-precision intrinsics

define float @add_f32(float %x, float %y) #0 {
; CHECK-LABEL: add_f32:
; CHECK: add.s
  %val = call float @llvm.experimental.constrained.fadd.f32(float %x, float %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define float @sub_f32(float %x, float %y) #0 {
; CHECK-LABEL: sub_f32:
; CHECK: sub.s
  %val = call float @llvm.experimental.constrained.fsub.f32(float %x, float %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define float @mul_f32(float %x, float %y) #0 {
; CHECK-LABEL: mul_f32:
; CHECK: mul.s
  %val = call float @llvm.experimental.constrained.fmul.f32(float %x, float %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define float @div_f32(float %x, float %y) #0 {
; CHECK-LABEL: div_f32:
; CHECK: div.s
  %val = call float @llvm.experimental.constrained.fdiv.f32(float %x, float %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define float @frem_f32(float %x, float %y) #0 {
; CHECK-LABEL: frem_f32:
; CHECK-R2: jal fmodf
; CHECK-R6: balc fmodf
  %val = call float @llvm.experimental.constrained.frem.f32(float %x, float %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define float @fma_f32(float %x, float %y, float %z) #0 {
; CHECK-LABEL: fma_f32:
; CHECK-R2: jal fmaf
; CHECK-R6: balc fmaf
  %val = call float @llvm.experimental.constrained.fma.f32(float %x, float %y, float %z, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define i32 @fptosi_f32(float %x) #0 {
; CHECK-LABEL: fptosi_f32:
; CHECK: trunc.w.s
  %val = call i32 @llvm.experimental.constrained.fptosi.i32.f32(float %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

define i32 @fptoui_f32(float %x) #0 {
; CHECK-LABEL: fptoui_f32:
; CHECK: trunc.w.s
; CHECK: trunc.w.s
  %val = call i32 @llvm.experimental.constrained.fptoui.i32.f32(float %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

define float @sqrt_f32(float %x) #0 {
; CHECK-LABEL: sqrt_f32:
; CHECK: sqrt.s
  %val = call float @llvm.experimental.constrained.sqrt.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define float @powi_f32(float %x, i32 %y) #0 {
; CHECK-LABEL: powi_f32:
; CHECK-R2: jal __powisf2
; CHECK-R6: balc __powisf2
  %val = call float @llvm.experimental.constrained.powi.f32(float %x, i32 %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define float @sin_f32(float %x) #0 {
; CHECK-LABEL: sin_f32:
; CHECK-R2: jal sinf
; CHECK-R6: balc sinf
  %val = call float @llvm.experimental.constrained.sin.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define float @cos_f32(float %x) #0 {
; CHECK-LABEL: cos_f32:
; CHECK-R2: jal cosf
; CHECK-R6: balc cosf
  %val = call float @llvm.experimental.constrained.cos.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define float @tan_f32(float %x) #0 {
; CHECK-LABEL: tan_f32:
; CHECK-R2: jal tanf
; CHECK-R6: balc tanf
  %val = call float @llvm.experimental.constrained.tan.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define float @acos_f32(float %x, float %y) #0 {
; CHECK-LABEL: acos_f32:
; CHECK-R2: jal acosf
; CHECK-R6: balc acosf
  %val = call float @llvm.experimental.constrained.acos.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define float @asin_f32(float %x, float %y) #0 {
; CHECK-LABEL: asin_f32:
; CHECK-R2: jal asinf
; CHECK-R6: balc asinf
  %val = call float @llvm.experimental.constrained.asin.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define float @atan_f32(float %x, float %y) #0 {
; CHECK-LABEL: atan_f32:
; CHECK-R2: jal atanf
; CHECK-R6: balc atanf
  %val = call float @llvm.experimental.constrained.atan.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define float @cosh_f32(float %x, float %y) #0 {
; CHECK-LABEL: cosh_f32:
; CHECK-R2: jal coshf
; CHECK-R6: balc coshf
  %val = call float @llvm.experimental.constrained.cosh.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define float @sinh_f32(float %x, float %y) #0 {
; CHECK-LABEL: sinh_f32:
; CHECK-R2: jal sinhf
; CHECK-R6: balc sinhf
  %val = call float @llvm.experimental.constrained.sinh.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define float @tanh_f32(float %x, float %y) #0 {
; CHECK-LABEL: tanh_f32:
; CHECK-R2: jal tanhf
; CHECK-R6: balc tanhf
  %val = call float @llvm.experimental.constrained.tanh.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define float @fmuladd_f32(float %x, float %y, float %z) #0 {
; CHECK-LABEL: fmuladd_f32:
; CHECK-R2: madd.s
; CHECK-R6: mul.s
; CHECK-R6: add.s
  %val = call float @llvm.experimental.constrained.fmuladd.f32(float %x, float %y, float %z, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define float @ldexp_f32(float %x, i32 %y) #0 {
; CHECK-LABEL: ldexp_f32:
; CHECK-R2: jal ldexpf
; CHECK-R6: balc ldexpf
  %val = call float @llvm.experimental.constrained.ldexp.f32.i32(float %x, i32 %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define float @roundeven_f32(float %x) #0 {
; CHECK-LABEL: roundeven_f32:
; CHECK-R2: jal roundevenf
; CHECK-R6: balc roundevenf
  %val = call float @llvm.experimental.constrained.roundeven.f32(float %x, metadata !"fpexcept.strict") #0
  ret float %val
}

define float @uitofp_f32_i32(i32 %x) #0 {
; CHECK-LABEL: uitofp_f32_i32:
; CHECK: lui
; CHECK: sw
; CHECK: mtc1
; CHECK: mthc1
; CHECK: ldc1
; CHECK: cvt.s.d
  %val = call float @llvm.experimental.constrained.uitofp.f32.i32(i32 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define float @atan2_f32(float %x, float %y) #0 {
; CHECK-LABEL: atan2_f32:
; CHECK-R2: jal atan2f
; CHECK-R6: balc atan2f
  %val = call float @llvm.experimental.constrained.atan2.f32(float %x, float %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define float @pow_f32(float %x, float %y) #0 {
; CHECK-LABEL: pow_f32:
; CHECK-R2: jal powf
; CHECK-R6: balc powf
  %val = call float @llvm.experimental.constrained.pow.f32(float %x, float %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define float @log_f32(float %x) #0 {
; CHECK-LABEL: log_f32:
; CHECK-R2: jal logf
; CHECK-R6: balc logf
  %val = call float @llvm.experimental.constrained.log.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define float @log10_f32(float %x) #0 {
; CHECK-LABEL: log10_f32:
; CHECK-R2: jal log10f
; CHECK-R6: balc log10f
  %val = call float @llvm.experimental.constrained.log10.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define float @log2_f32(float %x) #0 {
; CHECK-LABEL: log2_f32:
; CHECK-R2: jal log2f
; CHECK-R6: balc log2f
  %val = call float @llvm.experimental.constrained.log2.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define float @exp_f32(float %x) #0 {
; CHECK-LABEL: exp_f32:
; CHECK: jr $ra
  %val = call float @llvm.experimental.constrained.exp.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define float @exp2_f32(float %x) #0 {
; CHECK-LABEL: exp2_f32:
; CHECK-R2: jal exp2f
; CHECK-R6: balc exp2f
  %val = call float @llvm.experimental.constrained.exp2.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define float @rint_f32(float %x) #0 {
; CHECK-LABEL: rint_f32:
; CHECK-R2: jal rintf
<<<<<<< HEAD
; CHECK-R6: rint.s
=======
; CHECK-R6: balc rintf
>>>>>>> 544b04d01d3a ([Mips] Fix MipsJmpLink and MipsTailCall to use compact branch instructions on MIPS32/64 R6)
  %val = call float @llvm.experimental.constrained.rint.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define float @nearbyint_f32(float %x) #0 {
; CHECK-LABEL: nearbyint_f32:
; CHECK-R2: jal nearbyintf
; CHECK-R6: balc nearbyintf
  %val = call float @llvm.experimental.constrained.nearbyint.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define i32 @lrint_f32(float %x) #0 {
; CHECK-LABEL: lrint_f32:
; CHECK-R2: jal lrintf
; CHECK-R6: balc lrintf
  %val = call i32 @llvm.experimental.constrained.lrint.i32.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret i32 %val
}

define i32 @llrint_f32(float %x) #0 {
; CHECK-LABEL: llrint_f32:
; CHECK-R2: jal llrintf
; CHECK-R6: balc llrintf
  %val = call i32 @llvm.experimental.constrained.llrint.i32.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret i32 %val
}

define float @maxnum_f32(float %x, float %y) #0 {
; CHECK-LABEL: maxnum_f32:
; CHECK-R2: jal fmaxf
; CHECK-R6: max.s
  %val = call float @llvm.experimental.constrained.maxnum.f32(float %x, float %y, metadata !"fpexcept.strict") #0
  ret float %val
}

define float @minnum_f32(float %x, float %y) #0 {
; CHECK-LABEL: minnum_f32:
; CHECK-R2: jal fminf
; CHECK-R6: min.s
  %val = call float @llvm.experimental.constrained.minnum.f32(float %x, float %y, metadata !"fpexcept.strict") #0
  ret float %val
}

define float @ceil_f32(float %x) #0 {
; CHECK-LABEL: ceil_f32:
; CHECK-R2: jal ceilf
; CHECK-R6: balc ceilf
  %val = call float @llvm.experimental.constrained.ceil.f32(float %x, metadata !"fpexcept.strict") #0
  ret float %val
}

define float @floor_f32(float %x) #0 {
; CHECK-LABEL: floor_f32:
; CHECK-R2: jal floorf
; CHECK-R6: balc floorf
  %val = call float @llvm.experimental.constrained.floor.f32(float %x, metadata !"fpexcept.strict") #0
  ret float %val
}

define i32 @lround_f32(float %x) #0 {
; CHECK-LABEL: lround_f32:
; CHECK-R2: jal lroundf
; CHECK-R6: balc lroundf
  %val = call i32 @llvm.experimental.constrained.lround.i32.f32(float %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

define i32 @llround_f32(float %x) #0 {
; CHECK-LABEL: llround_f32:
; CHECK-R2: jal llroundf
; CHECK-R6: balc llroundf
  %val = call i32 @llvm.experimental.constrained.llround.i32.f32(float %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

define float @round_f32(float %x) #0 {
; CHECK-LABEL: round_f32:
; CHECK-R2: jal roundf
; CHECK-R6: balc roundf
  %val = call float @llvm.experimental.constrained.round.f32(float %x, metadata !"fpexcept.strict") #0
  ret float %val
}

define float @trunc_f32(float %x) #0 {
; CHECK-LABEL: trunc_f32:
; CHECK-R2: jal truncf
; CHECK-R6: balc truncf
  %val = call float @llvm.experimental.constrained.trunc.f32(float %x, metadata !"fpexcept.strict") #0
  ret float %val
}

; Double-precision intrinsics

define double @add_f64(double %x, double %y) #0 {
; CHECK-LABEL: add_f64:
; CHECK: add.d
  %val = call double @llvm.experimental.constrained.fadd.f64(double %x, double %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

define double @sub_f64(double %x, double %y) #0 {
; CHECK-LABEL: sub_f64:
; CHECK: sub.d
  %val = call double @llvm.experimental.constrained.fsub.f64(double %x, double %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

define double @mul_f64(double %x, double %y) #0 {
; CHECK-LABEL: mul_f64:
; CHECK: mul.d
  %val = call double @llvm.experimental.constrained.fmul.f64(double %x, double %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

define double @div_f64(double %x, double %y) #0 {
; CHECK-LABEL: div_f64:
; CHECK: div.d
  %val = call double @llvm.experimental.constrained.fdiv.f64(double %x, double %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

define double @frem_f64(double %x, double %y) #0 {
; CHECK-LABEL: frem_f64:
; CHECK-R2: jal fmod
; CHECK-R6: balc fmod
  %val = call double @llvm.experimental.constrained.frem.f64(double %x, double %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

define double @fma_f64(double %x, double %y, double %z) #0 {
; CHECK-LABEL: fma_f64:
; CHECK-R2: jal fma
; CHECK-R6: balc fma
  %val = call double @llvm.experimental.constrained.fma.f64(double %x, double %y, double %z, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

define i32 @fptosi_f64(double %x) #0 {
; CHECK-LABEL: fptosi_f64:
; CHECK: trunc.w.d
  %val = call i32 @llvm.experimental.constrained.fptosi.i32.f64(double %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

define i32 @fptoui_f64(double %x) #0 {
; CHECK-LABEL: fptoui_f64:
; CHECK: trunc.w.d 
; CHECK: trunc.w.d
  %val = call i32 @llvm.experimental.constrained.fptoui.i32.f64(double %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

define double @sqrt_f64(double %x) #0 {
; CHECK-LABEL: sqrt_f64:
; CHECK: sqrt.d
  %val = call double @llvm.experimental.constrained.sqrt.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

define double @powi_f64(double %x, i32 %y) #0 {
; CHECK-LABEL: powi_f64:
; CHECK-R2: jal __powidf2
; CHECK-R6: balc __powidf2
  %val = call double @llvm.experimental.constrained.powi.f64(double %x, i32 %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

define double @sin_f64(double %x) #0 {
; CHECK-LABEL: sin_f64:
; CHECK-R2: jal sin
; CHECK-R6: balc sin
  %val = call double @llvm.experimental.constrained.sin.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

define double @cos_f64(double %x) #0 {
; CHECK-LABEL: cos_f64:
; CHECK-R2: jal cos
; CHECK-R6: balc cos
  %val = call double @llvm.experimental.constrained.cos.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

define double @tan_f64(double %x) #0 {
; CHECK-LABEL: tan_f64:
; CHECK-R2: jal tan
; CHECK-R6: balc tan
  %val = call double @llvm.experimental.constrained.tan.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

define double @acos_f64(double %x, double %y) #0 {
; CHECK-LABEL: acos_f64:
; CHECK-R2: jal acos
; CHECK-R6: balc acos
  %val = call double @llvm.experimental.constrained.acos.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

define double @asin_f64(double %x, double %y) #0 {
; CHECK-LABEL: asin_f64:
; CHECK-R2: jal asin
; CHECK-R6: balc asin
  %val = call double @llvm.experimental.constrained.asin.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

define double @atan_f64(double %x, double %y) #0 {
; CHECK-LABEL: atan_f64:
; CHECK-R2: jal atan
; CHECK-R6: balc atan
  %val = call double @llvm.experimental.constrained.atan.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

define double @cosh_f64(double %x, double %y) #0 {
; CHECK-LABEL: cosh_f64:
; CHECK-R2: jal cosh
; CHECK-R6: balc cosh
  %val = call double @llvm.experimental.constrained.cosh.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

define double @sinh_f64(double %x, double %y) #0 {
; CHECK-LABEL: sinh_f64:
; CHECK-R2: jal sinh
; CHECK-R6: balc sinh
  %val = call double @llvm.experimental.constrained.sinh.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

define double @tanh_f64(double %x, double %y) #0 {
; CHECK-LABEL: tanh_f64:
; CHECK-R2: jal tanh
; CHECK-R6: balc tanh
  %val = call double @llvm.experimental.constrained.tanh.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

define double @fmuladd_f64(double %x, double %y, double %z) #0 {
; CHECK-LABEL: fmuladd_f64:
; CHECK-R2: madd.d
; CHECK-R6: mul.d
; CHECK-R6: add.d
  %val = call double @llvm.experimental.constrained.fmuladd.f64(double %x, double %y, double %z, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

define double @ldexp_f64(double %x, i32 %y) #0 {
; CHECK-LABEL: ldexp_f64:
; CHECK-R2: jal ldexp
; CHECK-R6: balc ldexp
  %val = call double @llvm.experimental.constrained.ldexp.f64.i32(double %x, i32 %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

define double @roundeven_f64(double %x) #0 {
; CHECK-LABEL: roundeven_f64:
; CHECK-R2: jal roundeven
; CHECK-R6: balc roundeven
  %val = call double @llvm.experimental.constrained.roundeven.f64(double %x, metadata !"fpexcept.strict") #0
  ret double %val
}

define double @uitofp_f64_i32(i32 %x) #0 {
; CHECK-LABEL: uitofp_f64_i32:
; CHECK: lui
; CHECK: sw
; CHECK: mtc1
; CHECK: mthc1
; CHECK: ldc1
  %val = call double @llvm.experimental.constrained.uitofp.f64.i32(i32 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

define double @atan2_f64(double %x, double %y) #0 {
; CHECK-LABEL: atan2_f64:
; CHECK-R2: jal atan2
; CHECK-R6: balc atan2
  %val = call double @llvm.experimental.constrained.atan2.f64(double %x, double %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

define double @pow_f64(double %x, double %y) #0 {
; CHECK-LABEL: pow_f64:
; CHECK-R2: jal pow
; CHECK-R6: balc pow
  %val = call double @llvm.experimental.constrained.pow.f64(double %x, double %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

define double @log_f64(double %x) #0 {
; CHECK-LABEL: log_f64:
; CHECK-R2: jal log
; CHECK-R6: balc log
  %val = call double @llvm.experimental.constrained.log.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

define double @log10_f64(double %x) #0 {
; CHECK-LABEL: log10_f64:
; CHECK-R2: jal log10
; CHECK-R6: balc log10
  %val = call double @llvm.experimental.constrained.log10.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

define double @log2_f64(double %x) #0 {
; CHECK-LABEL: log2_f64:
; CHECK-R2: jal log2
; CHECK-R6: balc log2
  %val = call double @llvm.experimental.constrained.log2.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

define double @exp_f64(double %x) #0 {
; CHECK-LABEL: exp_f64:
; CHECK-R2: jal exp
; CHECK-R6: balc exp
  %val = call double @llvm.experimental.constrained.exp.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

define double @exp2_f64(double %x) #0 {
; CHECK-LABEL: exp2_f64:
; CHECK-R2: jal exp2
; CHECK-R6: balc exp2
  %val = call double @llvm.experimental.constrained.exp2.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

define double @rint_f64(double %x) #0 {
; CHECK-LABEL: rint_f64:
; CHECK-R2: jal rint
; CHECK-R6: rint.d
  %val = call double @llvm.experimental.constrained.rint.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

define double @nearbyint_f64(double %x) #0 {
; CHECK-LABEL: nearbyint_f64:
; CHECK-R2: jal nearbyint
; CHECK-R6: balc nearbyint
  %val = call double @llvm.experimental.constrained.nearbyint.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

define i32 @lrint_f64(double %x) #0 {
; CHECK-LABEL: lrint_f64:
; CHECK-R2: jal lrint
; CHECK-R6: balc lrint
  %val = call i32 @llvm.experimental.constrained.lrint.i32.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret i32 %val
}

define i32 @llrint_f64(double %x) #0 {
; CHECK-LABEL: llrint_f64:
; CHECK-R2: jal llrint
; CHECK-R6: balc llrint
  %val = call i32 @llvm.experimental.constrained.llrint.i32.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret i32 %val
}

define double @maxnum_f64(double %x, double %y) #0 {
; CHECK-LABEL: maxnum_f64:
; CHECK-R2: jal fmax
; CHECK-R6: max.d
  %val = call double @llvm.experimental.constrained.maxnum.f64(double %x, double %y, metadata !"fpexcept.strict") #0
  ret double %val
}

define double @minnum_f64(double %x, double %y) #0 {
; CHECK-LABEL: minnum_f64:
; CHECK-R2: jal fmin
; CHECK-R6: min.d
  %val = call double @llvm.experimental.constrained.minnum.f64(double %x, double %y, metadata !"fpexcept.strict") #0
  ret double %val
}

define double @ceil_f64(double %x) #0 {
; CHECK-LABEL: ceil_f64:
; CHECK-R2: jal ceil
; CHECK-R6: balc ceil
  %val = call double @llvm.experimental.constrained.ceil.f64(double %x, metadata !"fpexcept.strict") #0
  ret double %val
}

define double @floor_f64(double %x) #0 {
; CHECK-LABEL: floor_f64:
; CHECK-R2: jal floor
; CHECK-R6: balc floor
  %val = call double @llvm.experimental.constrained.floor.f64(double %x, metadata !"fpexcept.strict") #0
  ret double %val
}

define i32 @lround_f64(double %x) #0 {
; CHECK-LABEL: lround_f64:
; CHECK-R2: jal lround
; CHECK-R6: balc lround
  %val = call i32 @llvm.experimental.constrained.lround.i32.f64(double %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

define i32 @llround_f64(double %x) #0 {
; CHECK-LABEL: llround_f64:
; CHECK-R2: jal llround
; CHECK-R6: balc llround
  %val = call i32 @llvm.experimental.constrained.llround.i32.f64(double %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

define double @round_f64(double %x) #0 {
; CHECK-LABEL: round_f64:
; CHECK-R2: jal round
; CHECK-R6: balc round
  %val = call double @llvm.experimental.constrained.round.f64(double %x, metadata !"fpexcept.strict") #0
  ret double %val
}

define double @trunc_f64(double %x) #0 {
; CHECK-LABEL: trunc_f64:
; CHECK-R2: jal trunc
; CHECK-R6: balc trunc
  %val = call double @llvm.experimental.constrained.trunc.f64(double %x, metadata !"fpexcept.strict") #0
  ret double %val
}

define float @fptrunc_f32(double %x) #0 {
; CHECK-LABEL: fptrunc_f32:
; CHECK: cvt.s.d
  %val = call float @llvm.experimental.constrained.fptrunc.f32.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define double @fpext_f32(float %x) #0 {
; CHECK-LABEL: fpext_f32:
; CHECK: cvt.d.s
  %val = call double @llvm.experimental.constrained.fpext.f64.f32(float %x, metadata !"fpexcept.strict") #0
  ret double %val
}

define float @sitofp_f32_i32(i32 %x) #0 {
; CHECK-LABEL: sitofp_f32_i32:
; CHECK: ldc1
; CHECK: ldc1
; CHECK: cvt.s.d
  %val = call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

define double @sitofp_f64_i32(i32 %x) #0 {
; CHECK-LABEL: sitofp_f64_i32:
; CHECK: ldc1
; CHECK: ldc1
  %val = call double @llvm.experimental.constrained.sitofp.f64.i32(i32 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}


attributes #0 = { strictfp }

declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fsub.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fmul.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fdiv.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.frem.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fma.f32(float, float, float, metadata, metadata)
declare i32 @llvm.experimental.constrained.fptosi.i32.f32(float, metadata)
declare i32 @llvm.experimental.constrained.fptoui.i32.f32(float, metadata)
declare float @llvm.experimental.constrained.sqrt.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.powi.f32(float, i32, metadata, metadata)
declare float @llvm.experimental.constrained.sin.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.cos.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.tan.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.acos.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.asin.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.atan.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.cosh.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.sinh.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.tanh.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.fmuladd.f32(float, float, float, metadata, metadata)
declare float @llvm.experimental.constrained.ldexp.f32.i32(float, i32, metadata, metadata)
declare float @llvm.experimental.constrained.roundeven.f32(float, metadata)
declare float @llvm.experimental.constrained.uitofp.f32.i32(i32, metadata, metadata)
declare float @llvm.experimental.constrained.atan2.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.pow.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.log.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.log10.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.log2.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.exp.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.exp2.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.rint.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.nearbyint.f32(float, metadata, metadata)
declare i32 @llvm.experimental.constrained.lrint.i32.f32(float, metadata, metadata)
declare i32 @llvm.experimental.constrained.llrint.i32.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.maxnum.f32(float, float, metadata)
declare float @llvm.experimental.constrained.minnum.f32(float, float, metadata)
declare float @llvm.experimental.constrained.ceil.f32(float, metadata)
declare float @llvm.experimental.constrained.floor.f32(float, metadata)
declare i32 @llvm.experimental.constrained.lround.i32.f32(float, metadata)
declare i32 @llvm.experimental.constrained.llround.i32.f32(float, metadata)
declare float @llvm.experimental.constrained.round.f32(float, metadata)
declare float @llvm.experimental.constrained.trunc.f32(float, metadata)

declare double @llvm.experimental.constrained.fadd.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.fsub.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.fmul.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.fdiv.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.frem.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.fma.f64(double, double, double, metadata, metadata)
declare i32 @llvm.experimental.constrained.fptosi.i32.f64(double, metadata)
declare i32 @llvm.experimental.constrained.fptoui.i32.f64(double, metadata)
declare double @llvm.experimental.constrained.sqrt.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.powi.f64(double, i32, metadata, metadata)
declare double @llvm.experimental.constrained.sin.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.cos.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.tan.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.acos.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.asin.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.atan.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.cosh.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.sinh.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.tanh.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.fmuladd.f64(double, double, double, metadata, metadata)
declare double @llvm.experimental.constrained.ldexp.f64.i32(double, i32, metadata, metadata)
declare double @llvm.experimental.constrained.roundeven.f64(double, metadata)
declare double @llvm.experimental.constrained.uitofp.f64.i32(i32, metadata, metadata)
declare double @llvm.experimental.constrained.atan2.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.pow.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.log.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.log10.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.log2.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.exp.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.exp2.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.rint.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.nearbyint.f64(double, metadata, metadata)
declare i32 @llvm.experimental.constrained.lrint.i32.f64(double, metadata, metadata)
declare i32 @llvm.experimental.constrained.llrint.i32.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.maxnum.f64(double, double, metadata)
declare double @llvm.experimental.constrained.minnum.f64(double, double, metadata)
declare double @llvm.experimental.constrained.ceil.f64(double, metadata)
declare double @llvm.experimental.constrained.floor.f64(double, metadata)
declare i32 @llvm.experimental.constrained.lround.i32.f64(double, metadata)
declare i32 @llvm.experimental.constrained.llround.i32.f64(double, metadata)
declare double @llvm.experimental.constrained.round.f64(double, metadata)
declare double @llvm.experimental.constrained.trunc.f64(double, metadata)

declare float @llvm.experimental.constrained.fptrunc.f32.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.fpext.f64.f32(float, metadata)
declare float @llvm.experimental.constrained.sitofp.f32.i32(i32, metadata, metadata)
declare double @llvm.experimental.constrained.sitofp.f64.i32(i32, metadata, metadata)
