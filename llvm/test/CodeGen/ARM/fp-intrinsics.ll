; RUN: llc -mtriple=armv8a-none-eabi %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-SP,CHECK-DP,CHECK-SP-V8,CHECK-DP-V8
; RUN: llc -mtriple=thumbv8m.main-none-eabi %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NOSP,CHECK-NODP
; RUN: llc -mtriple=thumbv8m.main-none-eabi %s -o - -mattr=fp-armv8 | FileCheck %s --check-prefixes=CHECK,CHECK-SP,CHECK-DP,CHECK-SP-V8,CHECK-DP-V8
; RUN: llc -mtriple=thumbv8m.main-none-eabi %s -o - -mattr=fp-armv8sp | FileCheck %s --check-prefixes=CHECK,CHECK-SP,CHECK-NODP,CHECK-SP-V8
; RUN: llc -mtriple=armv7a-none-eabi %s -o - -mattr=vfp4 | FileCheck %s --check-prefixes=CHECK,CHECK-SP,CHECK-DP,CHECK-SP-NOV8,CHECK-DP-NOV8
; RUN: llc -mtriple=thumbv7m-none-eabi %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NOSP,CHECK-NODP
; RUN: llc -mtriple=thumbv7m-none-eabi %s -o - -mattr=vfp4 | FileCheck %s --check-prefixes=CHECK,CHECK-SP,CHECK-DP,CHECK-SP-NOV8,CHECK-DP-NOV8
; RUN: llc -mtriple=thumbv7m-none-eabi %s -o - -mattr=vfp4sp | FileCheck %s --check-prefixes=CHECK,CHECK-SP,CHECK-NODP,CHECK-SP-NOV8

; Check that constrained fp intrinsics are correctly lowered. In particular
; check that the valid combinations of single-precision and double-precision
; hardware being present or absent work as expected (i.e. we get an instruction
; when one is available, otherwise a libcall).

; FIXME: We're not generating the right instructions for some of these
; operations (see further FIXMEs down below).

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
  %val = call i32 @llvm.experimental.constrained.fptosi.i32.f32(float %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: fptosi_f32_twice:
; CHECK-NOSP: bl __aeabi_f2iz
; CHECK-NOSP: bl __aeabi_f2iz
; CHECK-SP: vcvt.s32.f32
; FIXME-CHECK-SP: vcvt.s32.f32
define void @fptosi_f32_twice(float %arg, ptr %ptr) #0 {
entry:
  %conv = call i32 @llvm.experimental.constrained.fptosi.i32.f32(float %arg, metadata !"fpexcept.strict") #0
  store i32 %conv, ptr %ptr, align 4
  %conv1 = call i32 @llvm.experimental.constrained.fptosi.i32.f32(float %arg, metadata !"fpexcept.strict") #0
  %idx = getelementptr inbounds i32, ptr %ptr, i32 1
  store i32 %conv1, ptr %idx, align 4
  ret void
}

; CHECK-LABEL: fptoui_f32:
; CHECK-NOSP: bl __aeabi_f2uiz
; FIXME-CHECK-SP: vcvt.u32.f32
define i32 @fptoui_f32(float %x) #0 {
  %val = call i32 @llvm.experimental.constrained.fptoui.i32.f32(float %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: fptoui_f32_twice:
; CHECK-NOSP: bl __aeabi_f2uiz
; CHECK-NOSP: bl __aeabi_f2uiz
; FIXME-CHECK-SP: vcvt.u32.f32
; FIXME-CHECK-SP: vcvt.u32.f32
define void @fptoui_f32_twice(float %arg, ptr %ptr) #0 {
entry:
  %conv = call i32 @llvm.experimental.constrained.fptoui.i32.f32(float %arg, metadata !"fpexcept.strict") #0
  store i32 %conv, ptr %ptr, align 4
  %conv1 = call i32 @llvm.experimental.constrained.fptoui.i32.f32(float %arg, metadata !"fpexcept.strict") #0
  %idx = getelementptr inbounds i32, ptr %ptr, i32 1
  store i32 %conv1, ptr %idx, align 4
  ret void
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

; CHECK-LABEL: tan_f32:
; CHECK: bl tanf
define float @tan_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.tan.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: atan2_f32:
; CHECK: bl atan2f
define float @atan2_f32(float %x, float %y) #0 {
  %val = call float @llvm.experimental.constrained.atan2.f32(float %x, float %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
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
; CHECK-SP-NOV8: bl rintf
; CHECK-SP-V8: vrintx.f32
define float @rint_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.rint.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: nearbyint_f32:
; CHECK-NOSP: bl nearbyintf
; CHECK-SP-NOV8: bl nearbyintf
; CHECK-SP-V8: vrintr.f32
define float @nearbyint_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.nearbyint.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: lrint_f32:
; CHECK: bl lrintf
define i32 @lrint_f32(float %x) #0 {
  %val = call i32 @llvm.experimental.constrained.lrint.i32.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: llrint_f32:
; CHECK: bl llrintf
define i32 @llrint_f32(float %x) #0 {
  %val = call i32 @llvm.experimental.constrained.llrint.i32.f32(float %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: maxnum_f32:
; CHECK-NOSP: bl fmaxf
; CHECK-SP-NOV8: bl fmaxf
; CHECK-SP-V8: vmaxnm.f32
define float @maxnum_f32(float %x, float %y) #0 {
  %val = call float @llvm.experimental.constrained.maxnum.f32(float %x, float %y, metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: minnum_f32:
; CHECK-NOSP: bl fminf
; CHECK-SP-NOV8: bl fminf
; CHECK-SP-V8: vminnm.f32
define float @minnum_f32(float %x, float %y) #0 {
  %val = call float @llvm.experimental.constrained.minnum.f32(float %x, float %y, metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: ceil_f32:
; CHECK-NOSP: bl ceilf
; CHECK-SP-NOV8: bl ceilf
; CHECK-SP-V8: vrintp.f32
define float @ceil_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.ceil.f32(float %x, metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: floor_f32:
; CHECK-NOSP: bl floorf
; CHECK-SP-NOV8: bl floorf
; CHECK-SP-V8: vrintm.f32
define float @floor_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.floor.f32(float %x, metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: lround_f32:
; CHECK: bl lroundf
define i32 @lround_f32(float %x) #0 {
  %val = call i32 @llvm.experimental.constrained.lround.i32.f32(float %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: llround_f32:
; CHECK: bl llroundf
define i32 @llround_f32(float %x) #0 {
  %val = call i32 @llvm.experimental.constrained.llround.i32.f32(float %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: round_f32:
; CHECK-NOSP: bl roundf
; CHECK-SP-NOV8: bl roundf
; CHECK-SP-V8: vrinta.f32
define float @round_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.round.f32(float %x, metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: trunc_f32:
; CHECK-NOSP: bl truncf
; CHECK-SP-NOV8: bl truncf
; CHECK-SP-V8: vrintz.f32
define float @trunc_f32(float %x) #0 {
  %val = call float @llvm.experimental.constrained.trunc.f32(float %x, metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: fcmp_olt_f32:
; CHECK-NOSP: bl __aeabi_fcmplt
; CHECK-SP: vcmp.f32
define i32 @fcmp_olt_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"olt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ole_f32:
; CHECK-NOSP: bl __aeabi_fcmple
; CHECK-SP: vcmp.f32
define i32 @fcmp_ole_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"ole", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ogt_f32:
; CHECK-NOSP: bl __aeabi_fcmpgt
; CHECK-SP: vcmp.f32
define i32 @fcmp_ogt_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"ogt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_oge_f32:
; CHECK-NOSP: bl __aeabi_fcmpge
; CHECK-SP: vcmp.f32
define i32 @fcmp_oge_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"oge", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_oeq_f32:
; CHECK-NOSP: bl __aeabi_fcmpeq
; CHECK-SP: vcmp.f32
define i32 @fcmp_oeq_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"oeq", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_one_f32:
; CHECK-NOSP: bl __aeabi_fcmpeq
; CHECK-NOSP: bl __aeabi_fcmpun
; CHECK-SP: vcmp.f32
define i32 @fcmp_one_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"one", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ult_f32:
; CHECK-NOSP: bl __aeabi_fcmpge
; CHECK-SP: vcmp.f32
define i32 @fcmp_ult_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"ult", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ule_f32:
; CHECK-NOSP: bl __aeabi_fcmpgt
; CHECK-SP: vcmp.f32
define i32 @fcmp_ule_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"ule", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ugt_f32:
; CHECK-NOSP: bl __aeabi_fcmple
; CHECK-SP: vcmp.f32
define i32 @fcmp_ugt_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"ugt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_uge_f32:
; CHECK-NOSP: bl __aeabi_fcmplt
; CHECK-SP: vcmp.f32
define i32 @fcmp_uge_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"uge", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ueq_f32:
; CHECK-NOSP: bl __aeabi_fcmpeq
; CHECK-NOSP: bl __aeabi_fcmpun
; CHECK-SP: vcmp.f32
define i32 @fcmp_ueq_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"ueq", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_une_f32:
; CHECK-NOSP: bl __aeabi_fcmpeq
; CHECK-SP: vcmp.f32
define i32 @fcmp_une_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"une", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_olt_f32:
; CHECK-NOSP: bl __aeabi_fcmplt
; CHECK-SP: vcmpe.f32
define i32 @fcmps_olt_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %b, metadata !"olt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ole_f32:
; CHECK-NOSP: bl __aeabi_fcmple
; CHECK-SP: vcmpe.f32
define i32 @fcmps_ole_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %b, metadata !"ole", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ogt_f32:
; CHECK-NOSP: bl __aeabi_fcmpgt
; CHECK-SP: vcmpe.f32
define i32 @fcmps_ogt_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %b, metadata !"ogt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_oge_f32:
; CHECK-NOSP: bl __aeabi_fcmpge
; CHECK-SP: vcmpe.f32
define i32 @fcmps_oge_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %b, metadata !"oge", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_oeq_f32:
; CHECK-NOSP: bl __aeabi_fcmpeq
; CHECK-SP: vcmpe.f32
define i32 @fcmps_oeq_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %b, metadata !"oeq", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_one_f32:
; CHECK-NOSP: bl __aeabi_fcmpeq
; CHECK-NOSP: bl __aeabi_fcmpun
; CHECK-SP: vcmpe.f32
define i32 @fcmps_one_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %b, metadata !"one", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ult_f32:
; CHECK-NOSP: bl __aeabi_fcmpge
; CHECK-SP: vcmpe.f32
define i32 @fcmps_ult_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %b, metadata !"ult", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ule_f32:
; CHECK-NOSP: bl __aeabi_fcmpgt
; CHECK-SP: vcmpe.f32
define i32 @fcmps_ule_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %b, metadata !"ule", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ugt_f32:
; CHECK-NOSP: bl __aeabi_fcmple
; CHECK-SP: vcmpe.f32
define i32 @fcmps_ugt_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %b, metadata !"ugt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_uge_f32:
; CHECK-NOSP: bl __aeabi_fcmplt
; CHECK-SP: vcmpe.f32
define i32 @fcmps_uge_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %b, metadata !"uge", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ueq_f32:
; CHECK-NOSP: bl __aeabi_fcmpeq
; CHECK-NOSP: bl __aeabi_fcmpun
; CHECK-SP: vcmpe.f32
define i32 @fcmps_ueq_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %b, metadata !"ueq", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_une_f32:
; CHECK-NOSP: bl __aeabi_fcmpeq
; CHECK-SP: vcmpe.f32
define i32 @fcmps_une_f32(float %a, float %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %b, metadata !"une", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
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
  %val = call i32 @llvm.experimental.constrained.fptosi.i32.f64(double %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: fptoui_f64:
; CHECK-NODP: bl __aeabi_d2uiz
; FIXME-CHECK-DP: vcvt.u32.f64
define i32 @fptoui_f64(double %x) #0 {
  %val = call i32 @llvm.experimental.constrained.fptoui.i32.f64(double %x, metadata !"fpexcept.strict") #0
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

; CHECK-LABEL: tan_f64:
; CHECK: bl tan
define double @tan_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.tan.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: atan2_f64:
; CHECK: bl atan2
define double @atan2_f64(double %x, double %y) #0 {
  %val = call double @llvm.experimental.constrained.atan2.f64(double %x, double %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
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
; CHECK-DP-NOV8: bl rint
; CHECK-DP-V8: vrintx.f64
define double @rint_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.rint.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: nearbyint_f64:
; CHECK-NODP: bl nearbyint
; CHECK-DP-NOV8: bl nearbyint
; CHECK-DP-V8: vrintr.f64
define double @nearbyint_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.nearbyint.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: lrint_f64:
; CHECK: bl lrint
define i32 @lrint_f64(double %x) #0 {
  %val = call i32 @llvm.experimental.constrained.lrint.i32.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: llrint_f64:
; CHECK: bl llrint
define i32 @llrint_f64(double %x) #0 {
  %val = call i32 @llvm.experimental.constrained.llrint.i32.f64(double %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: maxnum_f64:
; CHECK-NODP: bl fmax
; CHECK-DP-NOV8: bl fmax
; CHECK-DP-V8: vmaxnm.f64
define double @maxnum_f64(double %x, double %y) #0 {
  %val = call double @llvm.experimental.constrained.maxnum.f64(double %x, double %y, metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: minnum_f64:
; CHECK-NODP: bl fmin
; CHECK-DP-NOV8: bl fmin
; CHECK-DP-V8: vminnm.f64
define double @minnum_f64(double %x, double %y) #0 {
  %val = call double @llvm.experimental.constrained.minnum.f64(double %x, double %y, metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: ceil_f64:
; CHECK-NODP: bl ceil
; CHECK-DP-NOV8: bl ceil
; CHECK-DP-V8: vrintp.f64
define double @ceil_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.ceil.f64(double %x, metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: floor_f64:
; CHECK-NODP: bl floor
; CHECK-DP-NOV8: bl floor
; CHECK-DP-V8: vrintm.f64
define double @floor_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.floor.f64(double %x, metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: lround_f64:
; CHECK: bl lround
define i32 @lround_f64(double %x) #0 {
  %val = call i32 @llvm.experimental.constrained.lround.i32.f64(double %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: llround_f64:
; CHECK: bl llround
define i32 @llround_f64(double %x) #0 {
  %val = call i32 @llvm.experimental.constrained.llround.i32.f64(double %x, metadata !"fpexcept.strict") #0
  ret i32 %val
}

; CHECK-LABEL: round_f64:
; CHECK-NODP: bl round
; CHECK-DP-NOV8: bl round
; CHECK-DP-V8: vrinta.f64
define double @round_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.round.f64(double %x, metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: trunc_f64:
; CHECK-NODP: bl trunc
; CHECK-DP-NOV8: bl trunc
; CHECK-DP-V8: vrintz.f64
define double @trunc_f64(double %x) #0 {
  %val = call double @llvm.experimental.constrained.trunc.f64(double %x, metadata !"fpexcept.strict") #0
  ret double %val
}

; CHECK-LABEL: fcmp_olt_f64:
; CHECK-NODP: bl __aeabi_dcmplt
; CHECK-DP: vcmp.f64
define i32 @fcmp_olt_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"olt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ole_f64:
; CHECK-NODP: bl __aeabi_dcmple
; CHECK-DP: vcmp.f64
define i32 @fcmp_ole_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"ole", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ogt_f64:
; CHECK-NODP: bl __aeabi_dcmpgt
; CHECK-DP: vcmp.f64
define i32 @fcmp_ogt_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"ogt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_oge_f64:
; CHECK-NODP: bl __aeabi_dcmpge
; CHECK-DP: vcmp.f64
define i32 @fcmp_oge_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"oge", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_oeq_f64:
; CHECK-NODP: bl __aeabi_dcmpeq
; CHECK-DP: vcmp.f64
define i32 @fcmp_oeq_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"oeq", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_one_f64:
; CHECK-NODP-DAG: bl __aeabi_dcmpeq
; CHECK-NODP-DAG: bl __aeabi_dcmpun
; CHECK-DP: vcmp.f64
define i32 @fcmp_one_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"one", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ult_f64:
; CHECK-NODP: bl __aeabi_dcmpge
; CHECK-DP: vcmp.f64
define i32 @fcmp_ult_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"ult", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ule_f64:
; CHECK-NODP: bl __aeabi_dcmpgt
; CHECK-DP: vcmp.f64
define i32 @fcmp_ule_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"ule", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ugt_f64:
; CHECK-NODP: bl __aeabi_dcmple
; CHECK-DP: vcmp.f64
define i32 @fcmp_ugt_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"ugt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_uge_f64:
; CHECK-NODP: bl __aeabi_dcmplt
; CHECK-DP: vcmp.f64
define i32 @fcmp_uge_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"uge", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_ueq_f64:
; CHECK-NODP-DAG: bl __aeabi_dcmpeq
; CHECK-NODP-DAG: bl __aeabi_dcmpun
; CHECK-DP: vcmp.f64
define i32 @fcmp_ueq_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"ueq", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmp_une_f64:
; CHECK-NODP: bl __aeabi_dcmpeq
; CHECK-DP: vcmp.f64
define i32 @fcmp_une_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"une", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_olt_f64:
; CHECK-NODP: bl __aeabi_dcmplt
; CHECK-DP: vcmpe.f64
define i32 @fcmps_olt_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f64(double %a, double %b, metadata !"olt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ole_f64:
; CHECK-NODP: bl __aeabi_dcmple
; CHECK-DP: vcmpe.f64
define i32 @fcmps_ole_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f64(double %a, double %b, metadata !"ole", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ogt_f64:
; CHECK-NODP: bl __aeabi_dcmpgt
; CHECK-DP: vcmpe.f64
define i32 @fcmps_ogt_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f64(double %a, double %b, metadata !"ogt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_oge_f64:
; CHECK-NODP: bl __aeabi_dcmpge
; CHECK-DP: vcmpe.f64
define i32 @fcmps_oge_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f64(double %a, double %b, metadata !"oge", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_oeq_f64:
; CHECK-NODP: bl __aeabi_dcmpeq
; CHECK-DP: vcmpe.f64
define i32 @fcmps_oeq_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f64(double %a, double %b, metadata !"oeq", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_one_f64:
; CHECK-NODP-DAG: bl __aeabi_dcmpeq
; CHECK-NODP-DAG: bl __aeabi_dcmpun
; CHECK-DP: vcmpe.f64
define i32 @fcmps_one_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f64(double %a, double %b, metadata !"one", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ult_f64:
; CHECK-NODP: bl __aeabi_dcmpge
; CHECK-DP: vcmpe.f64
define i32 @fcmps_ult_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f64(double %a, double %b, metadata !"ult", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ule_f64:
; CHECK-NODP: bl __aeabi_dcmpgt
; CHECK-DP: vcmpe.f64
define i32 @fcmps_ule_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f64(double %a, double %b, metadata !"ule", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ugt_f64:
; CHECK-NODP: bl __aeabi_dcmple
; CHECK-DP: vcmpe.f64
define i32 @fcmps_ugt_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f64(double %a, double %b, metadata !"ugt", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_uge_f64:
; CHECK-NODP: bl __aeabi_dcmplt
; CHECK-DP: vcmpe.f64
define i32 @fcmps_uge_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f64(double %a, double %b, metadata !"uge", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_ueq_f64:
; CHECK-NODP-DAG: bl __aeabi_dcmpeq
; CHECK-NODP-DAG: bl __aeabi_dcmpun
; CHECK-DP: vcmpe.f64
define i32 @fcmps_ueq_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f64(double %a, double %b, metadata !"ueq", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: fcmps_une_f64:
; CHECK-NODP: bl __aeabi_dcmpeq
; CHECK-DP: vcmpe.f64
define i32 @fcmps_une_f64(double %a, double %b) #0 {
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f64(double %a, double %b, metadata !"une", metadata !"fpexcept.strict") #0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
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

; CHECK-LABEL: fpext_f32_twice:
; CHECK-NODP: bl __aeabi_f2d
; CHECK-NODP: bl __aeabi_f2d
; CHECK-DP: vcvt.f64.f32
; FIXME-CHECK-DP: vcvt.f64.f32
define void @fpext_f32_twice(float %arg, ptr %ptr) #0 {
entry:
  %conv1 = call double @llvm.experimental.constrained.fpext.f64.f32(float %arg, metadata !"fpexcept.strict") #0
  store double %conv1, ptr %ptr, align 8
  %conv2 = call double @llvm.experimental.constrained.fpext.f64.f32(float %arg, metadata !"fpexcept.strict") #0
  %idx = getelementptr inbounds double, ptr %ptr, i32 1
  store double %conv2, ptr %idx, align 8
  ret void
}

; CHECK-LABEL: sitofp_f32_i32:
; CHECK-NOSP: bl __aeabi_i2f
; FIXME-CHECK-SP: vcvt.f32.s32
define float @sitofp_f32_i32(i32 %x) #0 {
  %val = call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %val
}

; CHECK-LABEL: sitofp_f64_i32:
; FIXME-CHECK-NODP: bl __aeabi_i2d
; FIXME-CHECK-DP: vcvt.f64.s32
define double @sitofp_f64_i32(i32 %x) #0 {
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
declare double @llvm.experimental.constrained.sqrt.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.powi.f64(double, i32, metadata, metadata)
declare double @llvm.experimental.constrained.sin.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.cos.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.tan.f64(double, metadata, metadata)
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
declare i1 @llvm.experimental.constrained.fcmps.f64(double, double, metadata, metadata)
declare i1 @llvm.experimental.constrained.fcmp.f64(double, double, metadata, metadata)

declare float @llvm.experimental.constrained.fptrunc.f32.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.fpext.f64.f32(float, metadata)
declare float @llvm.experimental.constrained.sitofp.f32.i32(i32, metadata, metadata)
declare double @llvm.experimental.constrained.sitofp.f64.i32(i32, metadata, metadata)
