; RUN: llc -mtriple=aarch64 %s -o - | FileCheck %s
; RUN: llc -mtriple=aarch64 -global-isel=true -global-isel-abort=2 %s -o - | FileCheck %s

; Check that constrained fp intrinsics are correctly lowered.


; Single-precision intrinsics

; CHECK-LABEL: add_f32:
; CHECK: fadd s0, s0, s1
define float @add_f32(float %x, float %y) #0 {
  %val = call float @llvm.experimental.constrained.fadd.f32(float %x, float %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: sub_f32:
; CHECK: fsub s0, s0, s1
define float @sub_f32(float %x, float %y) #0 {
  %val = call float @llvm.experimental.constrained.fsub.f32(float %x, float %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: mul_f32:
; CHECK: fmul s0, s0, s1
define float @mul_f32(float %x, float %y) #0 {
  %val = call float @llvm.experimental.constrained.fmul.f32(float %x, float %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: div_f32:
; CHECK: fdiv s0, s0, s1
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
; CHECK: fmadd s0, s0, s1, s2
define float @fma_f32(float %x, float %y, float %z) #0 {
  %val = call float @llvm.experimental.constrained.fma.f32(float %x, float %y, float %z, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: fptosi_i32_f32:
; CHECK: fcvtzs w0, s0
define i32 @fptosi_i32_f32(float %x) #0 {
  %val = call i32 @llvm.experimental.constrained.fptosi.i32.f32(float %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: fptoui_i32_f32:
; CHECK: fcvtzu w0, s0
define i32 @fptoui_i32_f32(float %x) #0 {
  %val = call i32 @llvm.experimental.constrained.fptoui.i32.f32(float %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: fptosi_i64_f32:
; CHECK: fcvtzs x0, s0
define i64 @fptosi_i64_f32(float %x) #0 {
  %val = call i64 @llvm.experimental.constrained.fptosi.i64.f32(float %x, metadata !"fpexcept.strict") #0
  ret i64 %val
}

; CHECK-LABEL: fptoui_i64_f32:
; CHECK: fcvtzu x0, s0
define i64 @fptoui_i64_f32(float %x) #0 {
  %val = call i64 @llvm.experimental.constrained.fptoui.i64.f32(float %x, metadata !"fpexcept.strict") #0
  ret i64 %val
}

; CHECK-LABEL: sitofp_f32_i32:
; CHECK: scvtf s0, w0
define float @sitofp_f32_i32(i32 %x) #0 {
  %val = call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: uitofp_f32_i32:
; CHECK: ucvtf s0, w0
define float @uitofp_f32_i32(i32 %x) #0 {
  %val = call float @llvm.experimental.constrained.uitofp.f32.i32(i32 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: sitofp_f32_i64:
; CHECK: scvtf s0, x0
define float @sitofp_f32_i64(i64 %x) #0 {
  %val = call float @llvm.experimental.constrained.sitofp.f32.i64(i64 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: uitofp_f32_i64:
; CHECK: ucvtf s0, x0
define float @uitofp_f32_i64(i64 %x) #0 {
  %val = call float @llvm.experimental.constrained.uitofp.f32.i64(i64 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: sitofp_f32_i128:
; CHECK: bl __floattisf
define float @sitofp_f32_i128(i128 %x) #0 {
  %val = call float @llvm.experimental.constrained.sitofp.f32.i128(i128 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: uitofp_f32_i128:
; CHECK: bl __floatuntisf
define float @uitofp_f32_i128(i128 %x) #0 {
  %val = call float @llvm.experimental.constrained.uitofp.f32.i128(i128 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: sqrt_f32:
; CHECK: fsqrt s0, s0
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

; CHECK-LABEL: tan_f32:
; CHECK: bl tanf
define float @tan_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.tan.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: asin_f32:
; CHECK: bl asinf
define float @asin_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.asin.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: acos_f32:
; CHECK: bl acosf
define float @acos_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.acos.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: atan_f32:
; CHECK: bl atanf
define float @atan_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.atan.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: sinh_f32:
; CHECK: bl sinhf
define float @sinh_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.sinh.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: cosh_f32:
; CHECK: bl coshf
define float @cosh_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.cosh.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: tanh_f32:
; CHECK: bl tanhf
define float @tanh_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.tanh.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
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
; CHECK: frintx s0, s0
define float @rint_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.rint.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: nearbyint_f32:
; CHECK: frinti s0, s0
define float @nearbyint_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.nearbyint.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: lrint_f32:
; CHECK: frintx [[REG:s[0-9]+]], s0
; CHECK: fcvtzs w0, [[REG]]
define i32 @lrint_f32(float %x) #0 {
  %val = call i32 @llvm.experimental.constrained.lrint.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: llrint_f32:
; CHECK: frintx [[REG:s[0-9]+]], s0
; CHECK: fcvtzs x0, [[REG]]
define i64 @llrint_f32(float %x) #0 {
  %val = call i64 @llvm.experimental.constrained.llrint.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret i64 %val
}

; CHECK-LABEL: maxnum_f32:
; CHECK: fmaxnm s0, s0, s1
define float @maxnum_f32(float %x, float %y) #0 {
  %val = call float @llvm.experimental.constrained.maxnum.f32(float %x, float %y, metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: minnum_f32:
; CHECK: fminnm s0, s0, s1
define float @minnum_f32(float %x, float %y) #0 {
  %val = call float @llvm.experimental.constrained.minnum.f32(float %x, float %y, metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: maximum_f32:
; CHECK: fmax s0, s0, s1
define float @maximum_f32(float %x, float %y) #0 {
  %val = call float @llvm.experimental.constrained.maximum.f32(float %x, float %y, metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: minimum_f32:
; CHECK: fmin s0, s0, s1
define float @minimum_f32(float %x, float %y) #0 {
  %val = call float @llvm.experimental.constrained.minimum.f32(float %x, float %y, metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: ceil_f32:
; CHECK: frintp s0, s0
define float @ceil_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.ceil.f32(float %x, metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: floor_f32:
; CHECK: frintm s0, s0
define float @floor_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.floor.f32(float %x, metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: lround_f32:
; CHECK: fcvtas w0, s0
define i32 @lround_f32(float %x) #0 {
  %val = call i32 @llvm.experimental.constrained.lround.f32(float %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: llround_f32:
; CHECK: fcvtas x0, s0
define i64 @llround_f32(float %x) #0 {
  %val = call i64 @llvm.experimental.constrained.llround.f32(float %x, metadata !"fpexcept.strict") #0
  ret i64 %val
}

; CHECK-LABEL: round_f32:
; CHECK: frinta s0, s0
define float @round_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.round.f32(float %x, metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: roundeven_f32:
; CHECK: frintn s0, s0
define float @roundeven_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.roundeven.f32(float %x, metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: trunc_f32:
; CHECK: frintz s0, s0
define float @trunc_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.trunc.f32(float %x, metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: fcmp_olt_f32:
; CHECK: fcmp s0, s1
define i32 @fcmp_olt_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"olt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ole_f32:
; CHECK: fcmp s0, s1
define i32 @fcmp_ole_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"ole", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ogt_f32:
; CHECK: fcmp s0, s1
define i32 @fcmp_ogt_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"ogt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_oge_f32:
; CHECK: fcmp s0, s1
define i32 @fcmp_oge_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"oge", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_oeq_f32:
; CHECK: fcmp s0, s1
define i32 @fcmp_oeq_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"oeq", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_one_f32:
; CHECK: fcmp s0, s1
define i32 @fcmp_one_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"one", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ult_f32:
; CHECK: fcmp s0, s1
define i32 @fcmp_ult_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"ult", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ule_f32:
; CHECK: fcmp s0, s1
define i32 @fcmp_ule_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"ule", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ugt_f32:
; CHECK: fcmp s0, s1
define i32 @fcmp_ugt_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"ugt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_uge_f32:
; CHECK: fcmp s0, s1
define i32 @fcmp_uge_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"uge", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ueq_f32:
; CHECK: fcmp s0, s1
define i32 @fcmp_ueq_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"ueq", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_une_f32:
; CHECK: fcmp s0, s1
define i32 @fcmp_une_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"une", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_olt_f32:
; CHECK: fcmpe s0, s1
define i32 @fcmps_olt_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %b, metadata !"olt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ole_f32:
; CHECK: fcmpe s0, s1
define i32 @fcmps_ole_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %b, metadata !"ole", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ogt_f32:
; CHECK: fcmpe s0, s1
define i32 @fcmps_ogt_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %b, metadata !"ogt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_oge_f32:
; CHECK: fcmpe s0, s1
define i32 @fcmps_oge_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %b, metadata !"oge", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_oeq_f32:
; CHECK: fcmpe s0, s1
define i32 @fcmps_oeq_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %b, metadata !"oeq", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_one_f32:
; CHECK: fcmpe s0, s1
define i32 @fcmps_one_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %b, metadata !"one", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ult_f32:
; CHECK: fcmpe s0, s1
define i32 @fcmps_ult_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %b, metadata !"ult", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ule_f32:
; CHECK: fcmpe s0, s1
define i32 @fcmps_ule_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %b, metadata !"ule", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ugt_f32:
; CHECK: fcmpe s0, s1
define i32 @fcmps_ugt_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %b, metadata !"ugt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_uge_f32:
; CHECK: fcmpe s0, s1
define i32 @fcmps_uge_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %b, metadata !"uge", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ueq_f32:
; CHECK: fcmpe s0, s1
define i32 @fcmps_ueq_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %b, metadata !"ueq", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_une_f32:
; CHECK: fcmpe s0, s1
define i32 @fcmps_une_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %b, metadata !"une", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}


; Double-precision intrinsics

; CHECK-LABEL: add_f64:
; CHECK: fadd d0, d0, d1
define double @add_f64(double %x, double %y) #0 {
  %val = call double @llvm.experimental.constrained.fadd.f64(double %x, double %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: sub_f64:
; CHECK: fsub d0, d0, d1
define double @sub_f64(double %x, double %y) #0 {
  %val = call double @llvm.experimental.constrained.fsub.f64(double %x, double %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: mul_f64:
; CHECK: fmul d0, d0, d1
define double @mul_f64(double %x, double %y) #0 {
  %val = call double @llvm.experimental.constrained.fmul.f64(double %x, double %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: div_f64:
; CHECK: fdiv d0, d0, d1
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
; CHECK: fmadd d0, d0, d1, d2
define double @fma_f64(double %x, double %y, double %z) #0 {
  %val = call double @llvm.experimental.constrained.fma.f64(double %x, double %y, double %z, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: fptosi_i32_f64:
; CHECK: fcvtzs w0, d0
define i32 @fptosi_i32_f64(double %x) #0 {
  %val = call i32 @llvm.experimental.constrained.fptosi.i32.f64(double %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: fptoui_i32_f64:
; CHECK: fcvtzu w0, d0
define i32 @fptoui_i32_f64(double %x) #0 {
  %val = call i32 @llvm.experimental.constrained.fptoui.i32.f64(double %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: fptosi_i64_f64:
; CHECK: fcvtzs x0, d0
define i64 @fptosi_i64_f64(double %x) #0 {
  %val = call i64 @llvm.experimental.constrained.fptosi.i64.f64(double %x, metadata !"fpexcept.strict") #0
  ret i64 %val
}

; CHECK-LABEL: fptoui_i64_f64:
; CHECK: fcvtzu x0, d0
define i64 @fptoui_i64_f64(double %x) #0 {
  %val = call i64 @llvm.experimental.constrained.fptoui.i64.f64(double %x, metadata !"fpexcept.strict") #0
  ret i64 %val
}

; CHECK-LABEL: sitofp_f64_i32:
; CHECK: scvtf d0, w0
define double @sitofp_f64_i32(i32 %x) #0 {
  %val = call double @llvm.experimental.constrained.sitofp.f64.i32(i32 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: uitofp_f64_i32:
; CHECK: ucvtf d0, w0
define double @uitofp_f64_i32(i32 %x) #0 {
  %val = call double @llvm.experimental.constrained.uitofp.f64.i32(i32 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: sitofp_f64_i64:
; CHECK: scvtf d0, x0
define double @sitofp_f64_i64(i64 %x) #0 {
  %val = call double @llvm.experimental.constrained.sitofp.f64.i64(i64 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: uitofp_f64_i64:
; CHECK: ucvtf d0, x0
define double @uitofp_f64_i64(i64 %x) #0 {
  %val = call double @llvm.experimental.constrained.uitofp.f64.i64(i64 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: sitofp_f64_i128:
; CHECK: bl __floattidf
define double @sitofp_f64_i128(i128 %x) #0 {
  %val = call double @llvm.experimental.constrained.sitofp.f64.i128(i128 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: uitofp_f64_i128:
; CHECK: bl __floatuntidf
define double @uitofp_f64_i128(i128 %x) #0 {
  %val = call double @llvm.experimental.constrained.uitofp.f64.i128(i128 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: sqrt_f64:
; CHECK: fsqrt d0, d0
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

; CHECK-LABEL: tan_f64:
; CHECK: bl tan
define double @tan_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.tan.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: asin_f64:
; CHECK: bl asin
define double @asin_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.asin.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: acos_f64:
; CHECK: bl acos
define double @acos_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.acos.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: atan_f64:
; CHECK: bl atan
define double @atan_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.atan.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: sinh_f64:
; CHECK: bl sinh
define double @sinh_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.sinh.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: cosh_f64:
; CHECK: bl cosh
define double @cosh_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.cosh.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: tanh_f64:
; CHECK: bl tanh
define double @tanh_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.tanh.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
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
; CHECK: frintx d0, d0
define double @rint_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.rint.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: nearbyint_f64:
; CHECK: frinti d0, d0
define double @nearbyint_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.nearbyint.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: lrint_f64:
; CHECK: frintx [[REG:d[0-9]+]], d0
; CHECK: fcvtzs w0, [[REG]]
define i32 @lrint_f64(double %x) #0 {
  %val = call i32 @llvm.experimental.constrained.lrint.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: llrint_f64:
; CHECK: frintx [[REG:d[0-9]+]], d0
; CHECK: fcvtzs x0, [[REG]]
define i64 @llrint_f64(double %x) #0 {
  %val = call i64 @llvm.experimental.constrained.llrint.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret i64 %val
}

; CHECK-LABEL: maxnum_f64:
; CHECK: fmaxnm d0, d0, d1
define double @maxnum_f64(double %x, double %y) #0 {
  %val = call double @llvm.experimental.constrained.maxnum.f64(double %x, double %y, metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: minnum_f64:
; CHECK: fminnm d0, d0, d1
define double @minnum_f64(double %x, double %y) #0 {
  %val = call double @llvm.experimental.constrained.minnum.f64(double %x, double %y, metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: maximum_f64:
; CHECK: fmax d0, d0, d1
define double @maximum_f64(double %x, double %y) #0 {
  %val = call double @llvm.experimental.constrained.maximum.f64(double %x, double %y, metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: minimum_f64:
; CHECK: fmin d0, d0, d1
define double @minimum_f64(double %x, double %y) #0 {
  %val = call double @llvm.experimental.constrained.minimum.f64(double %x, double %y, metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: ceil_f64:
; CHECK: frintp d0, d0
define double @ceil_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.ceil.f64(double %x, metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: floor_f64:
; CHECK: frintm d0, d0
define double @floor_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.floor.f64(double %x, metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: lround_f64:
; CHECK: fcvtas w0, d0
define i32 @lround_f64(double %x) #0 {
  %val = call i32 @llvm.experimental.constrained.lround.f64(double %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: llround_f64:
; CHECK: fcvtas x0, d0
define i64 @llround_f64(double %x) #0 {
  %val = call i64 @llvm.experimental.constrained.llround.f64(double %x, metadata !"fpexcept.strict") #0
  ret i64 %val
}

; CHECK-LABEL: round_f64:
; CHECK: frinta d0, d0
define double @round_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.round.f64(double %x, metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: roundeven_f64:
; CHECK: frintn d0, d0
define double @roundeven_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.roundeven.f64(double %x, metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: trunc_f64:
; CHECK: frintz d0, d0
define double @trunc_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.trunc.f64(double %x, metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: fcmp_olt_f64:
; CHECK: fcmp d0, d1
define i32 @fcmp_olt_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"olt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ole_f64:
; CHECK: fcmp d0, d1
define i32 @fcmp_ole_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"ole", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ogt_f64:
; CHECK: fcmp d0, d1
define i32 @fcmp_ogt_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"ogt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_oge_f64:
; CHECK: fcmp d0, d1
define i32 @fcmp_oge_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"oge", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_oeq_f64:
; CHECK: fcmp d0, d1
define i32 @fcmp_oeq_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"oeq", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_one_f64:
; CHECK: fcmp d0, d1
define i32 @fcmp_one_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"one", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ult_f64:
; CHECK: fcmp d0, d1
define i32 @fcmp_ult_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"ult", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ule_f64:
; CHECK: fcmp d0, d1
define i32 @fcmp_ule_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"ule", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ugt_f64:
; CHECK: fcmp d0, d1
define i32 @fcmp_ugt_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"ugt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_uge_f64:
; CHECK: fcmp d0, d1
define i32 @fcmp_uge_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"uge", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ueq_f64:
; CHECK: fcmp d0, d1
define i32 @fcmp_ueq_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"ueq", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_une_f64:
; CHECK: fcmp d0, d1
define i32 @fcmp_une_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"une", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_olt_f64:
; CHECK: fcmpe d0, d1
define i32 @fcmps_olt_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f64(double %a, double %b, metadata !"olt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ole_f64:
; CHECK: fcmpe d0, d1
define i32 @fcmps_ole_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f64(double %a, double %b, metadata !"ole", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ogt_f64:
; CHECK: fcmpe d0, d1
define i32 @fcmps_ogt_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f64(double %a, double %b, metadata !"ogt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_oge_f64:
; CHECK: fcmpe d0, d1
define i32 @fcmps_oge_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f64(double %a, double %b, metadata !"oge", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_oeq_f64:
; CHECK: fcmpe d0, d1
define i32 @fcmps_oeq_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f64(double %a, double %b, metadata !"oeq", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_one_f64:
; CHECK: fcmpe d0, d1
define i32 @fcmps_one_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f64(double %a, double %b, metadata !"one", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ult_f64:
; CHECK: fcmpe d0, d1
define i32 @fcmps_ult_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f64(double %a, double %b, metadata !"ult", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ule_f64:
; CHECK: fcmpe d0, d1
define i32 @fcmps_ule_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f64(double %a, double %b, metadata !"ule", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ugt_f64:
; CHECK: fcmpe d0, d1
define i32 @fcmps_ugt_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f64(double %a, double %b, metadata !"ugt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_uge_f64:
; CHECK: fcmpe d0, d1
define i32 @fcmps_uge_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f64(double %a, double %b, metadata !"uge", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ueq_f64:
; CHECK: fcmpe d0, d1
define i32 @fcmps_ueq_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f64(double %a, double %b, metadata !"ueq", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_une_f64:
; CHECK: fcmpe d0, d1
define i32 @fcmps_une_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f64(double %a, double %b, metadata !"une", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}


; Long-double-precision intrinsics

; CHECK-LABEL: add_f128:
; CHECK: bl __addtf3
define fp128 @add_f128(fp128 %x, fp128 %y) #0 {
  %val = call fp128 @llvm.experimental.constrained.fadd.f128(fp128 %x, fp128 %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: sub_f128:
; CHECK: bl __subtf3
define fp128 @sub_f128(fp128 %x, fp128 %y) #0 {
  %val = call fp128 @llvm.experimental.constrained.fsub.f128(fp128 %x, fp128 %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: mul_f128:
; CHECK: bl __multf3
define fp128 @mul_f128(fp128 %x, fp128 %y) #0 {
  %val = call fp128 @llvm.experimental.constrained.fmul.f128(fp128 %x, fp128 %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: div_f128:
; CHECK: bl __divtf3
define fp128 @div_f128(fp128 %x, fp128 %y) #0 {
  %val = call fp128 @llvm.experimental.constrained.fdiv.f128(fp128 %x, fp128 %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: frem_f128:
; CHECK: bl fmodl
define fp128 @frem_f128(fp128 %x, fp128 %y) #0 {
  %val = call fp128 @llvm.experimental.constrained.frem.f128(fp128 %x, fp128 %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: fma_f128:
; CHECK: fmal
define fp128 @fma_f128(fp128 %x, fp128 %y, fp128 %z) #0 {
  %val = call fp128 @llvm.experimental.constrained.fma.f128(fp128 %x, fp128 %y, fp128 %z, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: fptosi_i32_f128:
; CHECK: bl __fixtfsi
define i32 @fptosi_i32_f128(fp128 %x) #0 {
  %val = call i32 @llvm.experimental.constrained.fptosi.i32.f128(fp128 %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: fptoui_i32_f128:
; CHECK: bl __fixunstfsi
define i32 @fptoui_i32_f128(fp128 %x) #0 {
  %val = call i32 @llvm.experimental.constrained.fptoui.i32.f128(fp128 %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: fptosi_i64_f128:
; CHECK: bl __fixtfdi
define i64 @fptosi_i64_f128(fp128 %x) #0 {
  %val = call i64 @llvm.experimental.constrained.fptosi.i64.f128(fp128 %x, metadata !"fpexcept.strict") #0
  ret i64 %val
}

; CHECK-LABEL: fptoui_i64_f128:
; CHECK: bl __fixunstfdi
define i64 @fptoui_i64_f128(fp128 %x) #0 {
  %val = call i64 @llvm.experimental.constrained.fptoui.i64.f128(fp128 %x, metadata !"fpexcept.strict") #0
  ret i64 %val
}

; CHECK-LABEL: sitofp_f128_i32:
; CHECK: bl __floatsitf
define fp128 @sitofp_f128_i32(i32 %x) #0 {
  %val = call fp128 @llvm.experimental.constrained.sitofp.f128.i32(i32 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: uitofp_f128_i32:
; CHECK: bl __floatunsitf
define fp128 @uitofp_f128_i32(i32 %x) #0 {
  %val = call fp128 @llvm.experimental.constrained.uitofp.f128.i32(i32 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: sitofp_f128_i64:
; CHECK: bl __floatditf
define fp128 @sitofp_f128_i64(i64 %x) #0 {
  %val = call fp128 @llvm.experimental.constrained.sitofp.f128.i64(i64 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: uitofp_f128_i64:
; CHECK: bl __floatunditf
define fp128 @uitofp_f128_i64(i64 %x) #0 {
  %val = call fp128 @llvm.experimental.constrained.uitofp.f128.i64(i64 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: sitofp_f128_i128:
; CHECK: bl __floattitf
define fp128 @sitofp_f128_i128(i128 %x) #0 {
  %val = call fp128 @llvm.experimental.constrained.sitofp.f128.i128(i128 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: uitofp_f128_i128:
; CHECK: bl __floatuntitf
define fp128 @uitofp_f128_i128(i128 %x) #0 {
  %val = call fp128 @llvm.experimental.constrained.uitofp.f128.i128(i128 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: sqrt_f128:
; CHECK: bl sqrtl
define fp128 @sqrt_f128(fp128 %x) #0 {
  %val = call fp128 @llvm.experimental.constrained.sqrt.f128(fp128 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: powi_f128:
; CHECK: bl __powitf2
define fp128 @powi_f128(fp128 %x, i32 %y) #0 {
  %val = call fp128 @llvm.experimental.constrained.powi.f128(fp128 %x, i32 %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: sin_f128:
; CHECK: bl sinl
define fp128 @sin_f128(fp128 %x) #0 {
  %val = call fp128 @llvm.experimental.constrained.sin.f128(fp128 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: cos_f128:
; CHECK: bl cosl
define fp128 @cos_f128(fp128 %x) #0 {
  %val = call fp128 @llvm.experimental.constrained.cos.f128(fp128 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: tan_f128:
; CHECK: bl tanl
define fp128 @tan_f128(fp128 %x) #0 {
  %val = call fp128 @llvm.experimental.constrained.tan.f128(fp128 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: asin_f128:
; CHECK: bl asinl
define fp128 @asin_f128(fp128 %x) #0 {
  %val = call fp128 @llvm.experimental.constrained.asin.f128(fp128 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: acos_f128:
; CHECK: bl acosl
define fp128 @acos_f128(fp128 %x) #0 {
  %val = call fp128 @llvm.experimental.constrained.acos.f128(fp128 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: atan_f128:
; CHECK: bl atanl
define fp128 @atan_f128(fp128 %x) #0 {
  %val = call fp128 @llvm.experimental.constrained.atan.f128(fp128 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: sinh_f128:
; CHECK: bl sinhl
define fp128 @sinh_f128(fp128 %x) #0 {
  %val = call fp128 @llvm.experimental.constrained.sinh.f128(fp128 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: cosh_f128:
; CHECK: bl coshl
define fp128 @cosh_f128(fp128 %x) #0 {
  %val = call fp128 @llvm.experimental.constrained.cosh.f128(fp128 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: tanh_f128:
; CHECK: bl tanhl
define fp128 @tanh_f128(fp128 %x) #0 {
  %val = call fp128 @llvm.experimental.constrained.tanh.f128(fp128 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: pow_f128:
; CHECK: bl powl
define fp128 @pow_f128(fp128 %x, fp128 %y) #0 {
  %val = call fp128 @llvm.experimental.constrained.pow.f128(fp128 %x, fp128 %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: log_f128:
; CHECK: bl logl
define fp128 @log_f128(fp128 %x) #0 {
  %val = call fp128 @llvm.experimental.constrained.log.f128(fp128 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: log10_f128:
; CHECK: bl log10l
define fp128 @log10_f128(fp128 %x) #0 {
  %val = call fp128 @llvm.experimental.constrained.log10.f128(fp128 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: log2_f128:
; CHECK: bl log2l
define fp128 @log2_f128(fp128 %x) #0 {
  %val = call fp128 @llvm.experimental.constrained.log2.f128(fp128 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: exp_f128:
; CHECK: bl expl
define fp128 @exp_f128(fp128 %x) #0 {
  %val = call fp128 @llvm.experimental.constrained.exp.f128(fp128 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: exp2_f128:
; CHECK: bl exp2l
define fp128 @exp2_f128(fp128 %x) #0 {
  %val = call fp128 @llvm.experimental.constrained.exp2.f128(fp128 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: rint_f128:
; CHECK: bl rintl
define fp128 @rint_f128(fp128 %x) #0 {
  %val = call fp128 @llvm.experimental.constrained.rint.f128(fp128 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: nearbyint_f128:
; CHECK: bl nearbyintl
define fp128 @nearbyint_f128(fp128 %x) #0 {
  %val = call fp128 @llvm.experimental.constrained.nearbyint.f128(fp128 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: lrint_f128:
; CHECK: bl lrintl
define i32 @lrint_f128(fp128 %x) #0 {
  %val = call i32 @llvm.experimental.constrained.lrint.f128(fp128 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: llrint_f128:
; CHECK: bl llrintl
define i64 @llrint_f128(fp128 %x) #0 {
  %val = call i64 @llvm.experimental.constrained.llrint.f128(fp128 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret i64 %val
}

; CHECK-LABEL: maxnum_f128:
; CHECK: bl fmaxl
define fp128 @maxnum_f128(fp128 %x, fp128 %y) #0 {
  %val = call fp128 @llvm.experimental.constrained.maxnum.f128(fp128 %x, fp128 %y, metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: minnum_f128:
; CHECK: bl fminl
define fp128 @minnum_f128(fp128 %x, fp128 %y) #0 {
  %val = call fp128 @llvm.experimental.constrained.minnum.f128(fp128 %x, fp128 %y, metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: ceil_f128:
; CHECK: bl ceill
define fp128 @ceil_f128(fp128 %x) #0 {
  %val = call fp128 @llvm.experimental.constrained.ceil.f128(fp128 %x, metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: floor_f128:
; CHECK: bl floorl
define fp128 @floor_f128(fp128 %x) #0 {
  %val = call fp128 @llvm.experimental.constrained.floor.f128(fp128 %x, metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: lround_f128:
; CHECK: bl lroundl
define i32 @lround_f128(fp128 %x) #0 {
  %val = call i32 @llvm.experimental.constrained.lround.f128(fp128 %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: llround_f128:
; CHECK: bl llroundl
define i64 @llround_f128(fp128 %x) #0 {
  %val = call i64 @llvm.experimental.constrained.llround.f128(fp128 %x, metadata !"fpexcept.strict") #0
  ret i64 %val
}

; CHECK-LABEL: round_f128:
; CHECK: bl roundl
define fp128 @round_f128(fp128 %x) #0 {
  %val = call fp128 @llvm.experimental.constrained.round.f128(fp128 %x, metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: trunc_f128:
; CHECK: bl truncl
define fp128 @trunc_f128(fp128 %x) #0 {
  %val = call fp128 @llvm.experimental.constrained.trunc.f128(fp128 %x, metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: fcmp_olt_f128:
; CHECK: bl __lttf2
define i32 @fcmp_olt_f128(fp128 %a, fp128 %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f128(fp128 %a, fp128 %b, metadata !"olt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ole_f128:
; CHECK: bl __letf2
define i32 @fcmp_ole_f128(fp128 %a, fp128 %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f128(fp128 %a, fp128 %b, metadata !"ole", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ogt_f128:
; CHECK: bl __gttf2
define i32 @fcmp_ogt_f128(fp128 %a, fp128 %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f128(fp128 %a, fp128 %b, metadata !"ogt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_oge_f128:
; CHECK: bl __getf2
define i32 @fcmp_oge_f128(fp128 %a, fp128 %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f128(fp128 %a, fp128 %b, metadata !"oge", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_oeq_f128:
; CHECK: bl __eqtf2
define i32 @fcmp_oeq_f128(fp128 %a, fp128 %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f128(fp128 %a, fp128 %b, metadata !"oeq", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_one_f128:
; CHECK: bl __eqtf2
define i32 @fcmp_one_f128(fp128 %a, fp128 %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f128(fp128 %a, fp128 %b, metadata !"one", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ult_f128:
; CHECK: bl __getf2
define i32 @fcmp_ult_f128(fp128 %a, fp128 %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f128(fp128 %a, fp128 %b, metadata !"ult", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ule_f128:
; CHECK: bl __gttf2
define i32 @fcmp_ule_f128(fp128 %a, fp128 %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f128(fp128 %a, fp128 %b, metadata !"ule", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ugt_f128:
; CHECK: bl __letf2
define i32 @fcmp_ugt_f128(fp128 %a, fp128 %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f128(fp128 %a, fp128 %b, metadata !"ugt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_uge_f128:
; CHECK: bl __lttf2
define i32 @fcmp_uge_f128(fp128 %a, fp128 %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f128(fp128 %a, fp128 %b, metadata !"uge", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ueq_f128:
; CHECK: bl __eqtf2
define i32 @fcmp_ueq_f128(fp128 %a, fp128 %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f128(fp128 %a, fp128 %b, metadata !"ueq", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_une_f128:
; CHECK: bl __netf2
define i32 @fcmp_une_f128(fp128 %a, fp128 %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f128(fp128 %a, fp128 %b, metadata !"une", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_olt_f128:
; CHECK: bl __lttf2
define i32 @fcmps_olt_f128(fp128 %a, fp128 %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f128(fp128 %a, fp128 %b, metadata !"olt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ole_f128:
; CHECK: bl __letf2
define i32 @fcmps_ole_f128(fp128 %a, fp128 %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f128(fp128 %a, fp128 %b, metadata !"ole", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ogt_f128:
; CHECK: bl __gttf2
define i32 @fcmps_ogt_f128(fp128 %a, fp128 %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f128(fp128 %a, fp128 %b, metadata !"ogt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_oge_f128:
; CHECK: bl __getf2
define i32 @fcmps_oge_f128(fp128 %a, fp128 %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f128(fp128 %a, fp128 %b, metadata !"oge", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_oeq_f128:
; CHECK: bl __eqtf2
define i32 @fcmps_oeq_f128(fp128 %a, fp128 %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f128(fp128 %a, fp128 %b, metadata !"oeq", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_one_f128:
; CHECK: bl __eqtf2
define i32 @fcmps_one_f128(fp128 %a, fp128 %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f128(fp128 %a, fp128 %b, metadata !"one", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ult_f128:
; CHECK: bl __getf2
define i32 @fcmps_ult_f128(fp128 %a, fp128 %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f128(fp128 %a, fp128 %b, metadata !"ult", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ule_f128:
; CHECK: bl __gttf2
define i32 @fcmps_ule_f128(fp128 %a, fp128 %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f128(fp128 %a, fp128 %b, metadata !"ule", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ugt_f128:
; CHECK: bl __letf2
define i32 @fcmps_ugt_f128(fp128 %a, fp128 %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f128(fp128 %a, fp128 %b, metadata !"ugt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_uge_f128:
; CHECK: bl __lttf2
define i32 @fcmps_uge_f128(fp128 %a, fp128 %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f128(fp128 %a, fp128 %b, metadata !"uge", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ueq_f128:
; CHECK: bl __eqtf2
define i32 @fcmps_ueq_f128(fp128 %a, fp128 %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f128(fp128 %a, fp128 %b, metadata !"ueq", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_une_f128:
; CHECK: bl __netf2
define i32 @fcmps_une_f128(fp128 %a, fp128 %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f128(fp128 %a, fp128 %b, metadata !"une", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}


; Intrinsics to convert between floating-point types

; CHECK-LABEL: fptrunc_f32_f64:
; CHECK: fcvt s0, d0
define float @fptrunc_f32_f64(double %x) #0 {
  %val = call float @llvm.experimental.constrained.fptrunc.f32.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: fptrunc_f32_f128:
; CHECK: bl __trunctfsf2
define float @fptrunc_f32_f128(fp128 %x) #0 {
  %val = call float @llvm.experimental.constrained.fptrunc.f32.f128(fp128 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: fptrunc_f64_f128:
; CHECK: bl __trunctfdf2
define double @fptrunc_f64_f128(fp128 %x) #0 {
  %val = call double @llvm.experimental.constrained.fptrunc.f64.f128(fp128 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: fpext_f64_f32:
; CHECK: fcvt d0, s0
define double @fpext_f64_f32(float %x) #0 {
  %val = call double @llvm.experimental.constrained.fpext.f64.f32(float %x, metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: fpext_f128_f32:
; CHECK: bl __extendsftf2
define fp128 @fpext_f128_f32(float %x) #0 {
  %val = call fp128 @llvm.experimental.constrained.fpext.f128.f32(float %x, metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: fpext_f128_f64:
; CHECK: bl __extenddftf2
define fp128 @fpext_f128_f64(double %x) #0 {
  %val = call fp128 @llvm.experimental.constrained.fpext.f128.f64(double %x, metadata !"fpexcept.strict") #0
  ret fp128 %val
}

; CHECK-LABEL: sin_v1f64:
; CHECK: bl sin
define <1 x double> @sin_v1f64(<1 x double> %x, <1 x double> %y) #0 {
  %val = call <1 x double> @llvm.experimental.constrained.sin.v1f64(<1 x double> %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret <1 x double> %val
}

; CHECK-LABEL: cos_v1f64:
; CHECK: bl cos
define <1 x double> @cos_v1f64(<1 x double> %x, <1 x double> %y) #0 {
  %val = call <1 x double> @llvm.experimental.constrained.cos.v1f64(<1 x double> %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret <1 x double> %val
}

; CHECK-LABEL: tan_v1f64:
; CHECK: bl tan
define <1 x double> @tan_v1f64(<1 x double> %x, <1 x double> %y) #0 {
  %val = call <1 x double> @llvm.experimental.constrained.tan.v1f64(<1 x double> %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret <1 x double> %val
}

; CHECK-LABEL: asin_v1f64:
; CHECK: bl asin
define <1 x double> @asin_v1f64(<1 x double> %x, <1 x double> %y) #0 {
  %val = call <1 x double> @llvm.experimental.constrained.asin.v1f64(<1 x double> %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret <1 x double> %val
}

; CHECK-LABEL: acos_v1f64:
; CHECK: bl acos
define <1 x double> @acos_v1f64(<1 x double> %x, <1 x double> %y) #0 {
  %val = call <1 x double> @llvm.experimental.constrained.acos.v1f64(<1 x double> %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret <1 x double> %val
}

; CHECK-LABEL: atan_v1f64:
; CHECK: bl atan
define <1 x double> @atan_v1f64(<1 x double> %x, <1 x double> %y) #0 {
  %val = call <1 x double> @llvm.experimental.constrained.atan.v1f64(<1 x double> %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret <1 x double> %val
}

; CHECK-LABEL: sinh_v1f64:
; CHECK: bl sinh
define <1 x double> @sinh_v1f64(<1 x double> %x, <1 x double> %y) #0 {
  %val = call <1 x double> @llvm.experimental.constrained.sinh.v1f64(<1 x double> %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret <1 x double> %val
}

; CHECK-LABEL: cosh_v1f64:
; CHECK: bl cosh
define <1 x double> @cosh_v1f64(<1 x double> %x, <1 x double> %y) #0 {
  %val = call <1 x double> @llvm.experimental.constrained.cosh.v1f64(<1 x double> %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret <1 x double> %val
}

; CHECK-LABEL: tanh_v1f64:
; CHECK: bl tanh
define <1 x double> @tanh_v1f64(<1 x double> %x, <1 x double> %y) #0 {
  %val = call <1 x double> @llvm.experimental.constrained.tanh.v1f64(<1 x double> %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret <1 x double> %val
}

; CHECK-LABEL: pow_v1f64:
; CHECK: bl pow
define <1 x double> @pow_v1f64(<1 x double> %x, <1 x double> %y) #0 {
  %val = call <1 x double> @llvm.experimental.constrained.pow.v1f64(<1 x double> %x, <1 x double> %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret <1 x double> %val
}

; CHECK-LABEL: log_v1f64:
; CHECK: bl log
define <1 x double> @log_v1f64(<1 x double> %x, <1 x double> %y) #0 {
  %val = call <1 x double> @llvm.experimental.constrained.log.v1f64(<1 x double> %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret <1 x double> %val
}

; CHECK-LABEL: log2_v1f64:
; CHECK: bl log2
define <1 x double> @log2_v1f64(<1 x double> %x, <1 x double> %y) #0 {
  %val = call <1 x double> @llvm.experimental.constrained.log2.v1f64(<1 x double> %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret <1 x double> %val
}

; CHECK-LABEL: log10_v1f64:
; CHECK: bl log10
define <1 x double> @log10_v1f64(<1 x double> %x, <1 x double> %y) #0 {
  %val = call <1 x double> @llvm.experimental.constrained.log10.v1f64(<1 x double> %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret <1 x double> %val
}

; CHECK-LABEL: exp_v1f64:
; CHECK: bl exp
define <1 x double> @exp_v1f64(<1 x double> %x, <1 x double> %y) #0 {
  %val = call <1 x double> @llvm.experimental.constrained.exp.v1f64(<1 x double> %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret <1 x double> %val
}

; CHECK-LABEL: exp2_v1f64:
; CHECK: bl exp2
define <1 x double> @exp2_v1f64(<1 x double> %x, <1 x double> %y) #0 {
  %val = call <1 x double> @llvm.experimental.constrained.exp2.v1f64(<1 x double> %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret <1 x double> %val
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
declare i64 @llvm.experimental.constrained.fptosi.i64.f32(float, metadata)
declare i64 @llvm.experimental.constrained.fptoui.i64.f32(float, metadata)
declare float @llvm.experimental.constrained.sitofp.f32.i32(i32, metadata, metadata)
declare float @llvm.experimental.constrained.uitofp.f32.i32(i32, metadata, metadata)
declare float @llvm.experimental.constrained.sitofp.f32.i64(i64, metadata, metadata)
declare float @llvm.experimental.constrained.uitofp.f32.i64(i64, metadata, metadata)
declare float @llvm.experimental.constrained.sitofp.f32.i128(i128, metadata, metadata)
declare float @llvm.experimental.constrained.uitofp.f32.i128(i128, metadata, metadata)
declare float @llvm.experimental.constrained.sqrt.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.powi.f32(float, i32, metadata, metadata)
declare float @llvm.experimental.constrained.sin.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.cos.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.tan.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.asin.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.acos.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.atan.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.sinh.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.cosh.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.tanh.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.pow.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.log.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.log10.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.log2.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.exp.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.exp2.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.rint.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.nearbyint.f32(float, metadata, metadata)
declare i32 @llvm.experimental.constrained.lrint.f32(float, metadata, metadata)
declare i64 @llvm.experimental.constrained.llrint.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.maxnum.f32(float, float, metadata)
declare float @llvm.experimental.constrained.minnum.f32(float, float, metadata)
declare float @llvm.experimental.constrained.maximum.f32(float, float, metadata)
declare float @llvm.experimental.constrained.minimum.f32(float, float, metadata)
declare float @llvm.experimental.constrained.ceil.f32(float, metadata)
declare float @llvm.experimental.constrained.floor.f32(float, metadata)
declare i32 @llvm.experimental.constrained.lround.f32(float, metadata)
declare i64 @llvm.experimental.constrained.llround.f32(float, metadata)
declare float @llvm.experimental.constrained.round.f32(float, metadata)
declare float @llvm.experimental.constrained.roundeven.f32(float, metadata)
declare float @llvm.experimental.constrained.trunc.f32(float, metadata)
declare i1 @llvm.experimental.constrained.fcmps.f32(float, float, metadata, metadata)
declare i1 @llvm.experimental.constrained.fcmp.f32(float, float, metadata, metadata)

declare double @llvm.experimental.constrained.fadd.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.fsub.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.fmul.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.fdiv.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.frem.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.fma.f64(double, double, double, metadata, metadata)
declare i32 @llvm.experimental.constrained.fptosi.i32.f64(double, metadata)
declare i32 @llvm.experimental.constrained.fptoui.i32.f64(double, metadata)
declare i64 @llvm.experimental.constrained.fptosi.i64.f64(double, metadata)
declare i64 @llvm.experimental.constrained.fptoui.i64.f64(double, metadata)
declare double @llvm.experimental.constrained.sitofp.f64.i32(i32, metadata, metadata)
declare double @llvm.experimental.constrained.uitofp.f64.i32(i32, metadata, metadata)
declare double @llvm.experimental.constrained.sitofp.f64.i64(i64, metadata, metadata)
declare double @llvm.experimental.constrained.uitofp.f64.i64(i64, metadata, metadata)
declare double @llvm.experimental.constrained.sitofp.f64.i128(i128, metadata, metadata)
declare double @llvm.experimental.constrained.uitofp.f64.i128(i128, metadata, metadata)
declare double @llvm.experimental.constrained.sqrt.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.powi.f64(double, i32, metadata, metadata)
declare double @llvm.experimental.constrained.sin.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.cos.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.tan.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.asin.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.acos.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.atan.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.sinh.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.cosh.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.tanh.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.pow.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.log.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.log10.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.log2.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.exp.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.exp2.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.rint.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.nearbyint.f64(double, metadata, metadata)
declare i32 @llvm.experimental.constrained.lrint.f64(double, metadata, metadata)
declare i64 @llvm.experimental.constrained.llrint.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.maxnum.f64(double, double, metadata)
declare double @llvm.experimental.constrained.minnum.f64(double, double, metadata)
declare double @llvm.experimental.constrained.maximum.f64(double, double, metadata)
declare double @llvm.experimental.constrained.minimum.f64(double, double, metadata)
declare double @llvm.experimental.constrained.ceil.f64(double, metadata)
declare double @llvm.experimental.constrained.floor.f64(double, metadata)
declare i32 @llvm.experimental.constrained.lround.f64(double, metadata)
declare i64 @llvm.experimental.constrained.llround.f64(double, metadata)
declare double @llvm.experimental.constrained.round.f64(double, metadata)
declare double @llvm.experimental.constrained.roundeven.f64(double, metadata)
declare double @llvm.experimental.constrained.trunc.f64(double, metadata)
declare i1 @llvm.experimental.constrained.fcmps.f64(double, double, metadata, metadata)
declare i1 @llvm.experimental.constrained.fcmp.f64(double, double, metadata, metadata)

declare fp128 @llvm.experimental.constrained.fadd.f128(fp128, fp128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.fsub.f128(fp128, fp128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.fmul.f128(fp128, fp128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.fdiv.f128(fp128, fp128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.frem.f128(fp128, fp128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.fma.f128(fp128, fp128, fp128, metadata, metadata)
declare i32 @llvm.experimental.constrained.fptosi.i32.f128(fp128, metadata)
declare i32 @llvm.experimental.constrained.fptoui.i32.f128(fp128, metadata)
declare i64 @llvm.experimental.constrained.fptosi.i64.f128(fp128, metadata)
declare i64 @llvm.experimental.constrained.fptoui.i64.f128(fp128, metadata)
declare fp128 @llvm.experimental.constrained.sitofp.f128.i32(i32, metadata, metadata)
declare fp128 @llvm.experimental.constrained.uitofp.f128.i32(i32, metadata, metadata)
declare fp128 @llvm.experimental.constrained.sitofp.f128.i64(i64, metadata, metadata)
declare fp128 @llvm.experimental.constrained.uitofp.f128.i64(i64, metadata, metadata)
declare fp128 @llvm.experimental.constrained.sitofp.f128.i128(i128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.uitofp.f128.i128(i128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.sqrt.f128(fp128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.powi.f128(fp128, i32, metadata, metadata)
declare fp128 @llvm.experimental.constrained.sin.f128(fp128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.cos.f128(fp128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.tan.f128(fp128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.asin.f128(fp128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.acos.f128(fp128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.atan.f128(fp128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.sinh.f128(fp128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.cosh.f128(fp128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.tanh.f128(fp128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.pow.f128(fp128, fp128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.log.f128(fp128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.log10.f128(fp128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.log2.f128(fp128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.exp.f128(fp128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.exp2.f128(fp128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.rint.f128(fp128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.nearbyint.f128(fp128, metadata, metadata)
declare i32 @llvm.experimental.constrained.lrint.f128(fp128, metadata, metadata)
declare i64 @llvm.experimental.constrained.llrint.f128(fp128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.maxnum.f128(fp128, fp128, metadata)
declare fp128 @llvm.experimental.constrained.minnum.f128(fp128, fp128, metadata)
declare fp128 @llvm.experimental.constrained.ceil.f128(fp128, metadata)
declare fp128 @llvm.experimental.constrained.floor.f128(fp128, metadata)
declare i32 @llvm.experimental.constrained.lround.f128(fp128, metadata)
declare i64 @llvm.experimental.constrained.llround.f128(fp128, metadata)
declare fp128 @llvm.experimental.constrained.round.f128(fp128, metadata)
declare fp128 @llvm.experimental.constrained.trunc.f128(fp128, metadata)
declare i1 @llvm.experimental.constrained.fcmps.f128(fp128, fp128, metadata, metadata)
declare i1 @llvm.experimental.constrained.fcmp.f128(fp128, fp128, metadata, metadata)

declare float @llvm.experimental.constrained.fptrunc.f32.f64(double, metadata, metadata)
declare float @llvm.experimental.constrained.fptrunc.f32.f128(fp128, metadata, metadata)
declare double @llvm.experimental.constrained.fptrunc.f64.f128(fp128, metadata, metadata)
declare double @llvm.experimental.constrained.fpext.f64.f32(float, metadata)
declare fp128 @llvm.experimental.constrained.fpext.f128.f32(float, metadata)
declare fp128 @llvm.experimental.constrained.fpext.f128.f64(double, metadata)
