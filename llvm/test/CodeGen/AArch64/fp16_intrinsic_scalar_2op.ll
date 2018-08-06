; RUN: llc < %s -mtriple=aarch64-eabi -mattr=+v8.2a,+fullfp16  | FileCheck %s

declare half @llvm.aarch64.sisd.fabd.f16(half, half)
declare half @llvm.aarch64.neon.fmax.f16(half, half)
declare half @llvm.aarch64.neon.fmin.f16(half, half)
declare half @llvm.aarch64.neon.frsqrts.f16(half, half)
declare half @llvm.aarch64.neon.frecps.f16(half, half)
declare half @llvm.aarch64.neon.fmulx.f16(half, half)
declare half @llvm.fabs.f16(half)

define dso_local half @t_vabdh_f16(half %a, half %b) {
; CHECK-LABEL: t_vabdh_f16:
; CHECK:         fabd h0, h0, h1
; CHECK-NEXT:    ret
entry:
  %vabdh_f16 = tail call half @llvm.aarch64.sisd.fabd.f16(half %a, half %b)
  ret half %vabdh_f16
}

define dso_local half @t_vabdh_f16_from_fsub_fabs(half %a, half %b) {
; CHECK-LABEL: t_vabdh_f16_from_fsub_fabs:
; CHECK:         fabd h0, h0, h1
; CHECK-NEXT:    ret
entry:
  %sub = fsub half %a, %b
  %abs = tail call half @llvm.fabs.f16(half %sub)
  ret half %abs
}

define dso_local i16 @t_vceqh_f16(half %a, half %b) {
; CHECK-LABEL: t_vceqh_f16:
; CHECK:         fcmp h0, h1
; CHECK-NEXT:    csetm w0, eq
; CHECK-NEXT:    ret
entry:
  %0 = fcmp oeq half %a, %b
  %vcmpd = sext i1 %0 to i16
  ret i16 %vcmpd
}

define dso_local i16 @t_vcgeh_f16(half %a, half %b) {
; CHECK-LABEL: t_vcgeh_f16:
; CHECK:         fcmp h0, h1
; CHECK-NEXT:    csetm w0, ge
; CHECK-NEXT:    ret
entry:
  %0 = fcmp oge half %a, %b
  %vcmpd = sext i1 %0 to i16
  ret i16 %vcmpd
}

define dso_local i16 @t_vcgth_f16(half %a, half %b) {
; CHECK-LABEL: t_vcgth_f16:
; CHECK:         fcmp h0, h1
; CHECK-NEXT:    csetm w0, gt
; CHECK-NEXT:    ret
entry:
  %0 = fcmp ogt half %a, %b
  %vcmpd = sext i1 %0 to i16
  ret i16 %vcmpd
}

define dso_local i16 @t_vcleh_f16(half %a, half %b) {
; CHECK-LABEL: t_vcleh_f16:
; CHECK:         fcmp h0, h1
; CHECK-NEXT:    csetm w0, ls
; CHECK-NEXT:    ret
entry:
  %0 = fcmp ole half %a, %b
  %vcmpd = sext i1 %0 to i16
  ret i16 %vcmpd
}

define dso_local i16 @t_vclth_f16(half %a, half %b) {
; CHECK-LABEL: t_vclth_f16:
; CHECK:         fcmp h0, h1
; CHECK-NEXT:    csetm w0, mi
; CHECK-NEXT:    ret
entry:
  %0 = fcmp olt half %a, %b
  %vcmpd = sext i1 %0 to i16
  ret i16 %vcmpd
}

define dso_local half @t_vmaxh_f16(half %a, half %b) {
; CHECK-LABEL: t_vmaxh_f16:
; CHECK:         fmax h0, h0, h1
; CHECK-NEXT:    ret
entry:
  %vmax = tail call half @llvm.aarch64.neon.fmax.f16(half %a, half %b)
  ret half %vmax
}

define dso_local half @t_vminh_f16(half %a, half %b) {
; CHECK-LABEL: t_vminh_f16:
; CHECK:         fmin h0, h0, h1
; CHECK-NEXT:    ret
entry:
  %vmin = tail call half @llvm.aarch64.neon.fmin.f16(half %a, half %b)
  ret half %vmin
}

define dso_local half @t_vmulxh_f16(half %a, half %b) {
; CHECK-LABEL: t_vmulxh_f16:
; CHECK:         fmulx h0, h0, h1
; CHECK-NEXT:    ret
entry:
  %vmulxh_f16 = tail call half @llvm.aarch64.neon.fmulx.f16(half %a, half %b)
  ret half %vmulxh_f16
}

define dso_local half @t_vrecpsh_f16(half %a, half %b) {
; CHECK-LABEL: t_vrecpsh_f16:
; CHECK:         frecps h0, h0, h1
; CHECK-NEXT:    ret
entry:
  %vrecps = tail call half @llvm.aarch64.neon.frecps.f16(half %a, half %b)
  ret half %vrecps
}

define dso_local half @t_vrsqrtsh_f16(half %a, half %b) {
; CHECK-LABEL: t_vrsqrtsh_f16:
; CHECK:         frsqrts h0, h0, h1
; CHECK-NEXT:    ret
entry:
  %vrsqrtsh_f16 = tail call half @llvm.aarch64.neon.frsqrts.f16(half %a, half %b)
  ret half %vrsqrtsh_f16
}

declare half @llvm.aarch64.neon.vcvtfxs2fp.f16.i32(i32, i32) #1
declare half @llvm.aarch64.neon.vcvtfxs2fp.f16.i64(i64, i32) #1
declare i32 @llvm.aarch64.neon.vcvtfp2fxs.i32.f16(half, i32) #1
declare i64 @llvm.aarch64.neon.vcvtfp2fxs.i64.f16(half, i32) #1
declare half @llvm.aarch64.neon.vcvtfxu2fp.f16.i32(i32, i32) #1
declare i32 @llvm.aarch64.neon.vcvtfp2fxu.i32.f16(half, i32) #1

define dso_local half @test_vcvth_n_f16_s16_1(i16 %a) {
; CHECK-LABEL: test_vcvth_n_f16_s16_1:
; CHECK:         fmov s0, w[[wReg:[0-9]+]]
; CHECK-NEXT:    scvtf h0, h0, #1
; CHECK-NEXT:    ret
entry:
  %sext = sext i16 %a to i32
  %fcvth_n = tail call half @llvm.aarch64.neon.vcvtfxs2fp.f16.i32(i32 %sext, i32 1)
  ret half %fcvth_n
}

define dso_local half @test_vcvth_n_f16_s16_16(i16 %a) {
; CHECK-LABEL: test_vcvth_n_f16_s16_16:
; CHECK:         fmov s0, w[[wReg:[0-9]+]]
; CHECK-NEXT:    scvtf h0, h0, #16
; CHECK-NEXT:    ret
entry:
  %sext = sext i16 %a to i32
  %fcvth_n = tail call half @llvm.aarch64.neon.vcvtfxs2fp.f16.i32(i32 %sext, i32 16)
  ret half %fcvth_n
}

define dso_local half @test_vcvth_n_f16_s32_1(i32 %a) {
; CHECK-LABEL: test_vcvth_n_f16_s32_1:
; CHECK:         fmov s0, w0
; CHECK-NEXT:    scvtf h0, h0, #1
; CHECK-NEXT:    ret
entry:
  %vcvth_n_f16_s32 = tail call half @llvm.aarch64.neon.vcvtfxs2fp.f16.i32(i32 %a, i32 1)
  ret half %vcvth_n_f16_s32
}

define dso_local half @test_vcvth_n_f16_s32_16(i32 %a) {
; CHECK-LABEL: test_vcvth_n_f16_s32_16:
; CHECK:         fmov s0, w0
; CHECK-NEXT:    scvtf h0, h0, #16
; CHECK-NEXT:    ret
entry:
  %vcvth_n_f16_s32 = tail call half @llvm.aarch64.neon.vcvtfxs2fp.f16.i32(i32 %a, i32 16)
  ret half %vcvth_n_f16_s32
}

define dso_local i16 @test_vcvth_n_s16_f16_1(half %a) {
; CHECK-LABEL: test_vcvth_n_s16_f16_1:
; CHECK:         fcvtzs h0, h0, #1
; CHECK-NEXT:    fmov w0, s0
; CHECK-NEXT:    ret
entry:
  %fcvth_n = tail call i32 @llvm.aarch64.neon.vcvtfp2fxs.i32.f16(half %a, i32 1)
  %0 = trunc i32 %fcvth_n to i16
  ret i16 %0
}

define dso_local i16 @test_vcvth_n_s16_f16_16(half %a) {
; CHECK-LABEL: test_vcvth_n_s16_f16_16:
; CHECK:         fcvtzs h0, h0, #16
; CHECK-NEXT:    fmov w0, s0
; CHECK-NEXT:    ret
entry:
  %fcvth_n = tail call i32 @llvm.aarch64.neon.vcvtfp2fxs.i32.f16(half %a, i32 16)
  %0 = trunc i32 %fcvth_n to i16
  ret i16 %0
}

define dso_local i32 @test_vcvth_n_s32_f16_1(half %a) {
; CHECK-LABEL: test_vcvth_n_s32_f16_1:
; CHECK:         fcvtzs h0, h0, #1
; CHECK-NEXT:    fmov w0, s0
; CHECK-NEXT:    ret
entry:
  %vcvth_n_s32_f16 = tail call i32 @llvm.aarch64.neon.vcvtfp2fxs.i32.f16(half %a, i32 1)
  ret i32 %vcvth_n_s32_f16
}

define dso_local i32 @test_vcvth_n_s32_f16_16(half %a) {
; CHECK-LABEL: test_vcvth_n_s32_f16_16:
; CHECK:         fcvtzs h0, h0, #16
; CHECK-NEXT:    fmov w0, s0
; CHECK-NEXT:    ret
entry:
  %vcvth_n_s32_f16 = tail call i32 @llvm.aarch64.neon.vcvtfp2fxs.i32.f16(half %a, i32 16)
  ret i32 %vcvth_n_s32_f16
}

define dso_local i64 @test_vcvth_n_s64_f16_1(half %a) {
; CHECK-LABEL: test_vcvth_n_s64_f16_1:
; CHECK:         fcvtzs h0, h0, #1
; CHECK-NEXT:    fmov x0, d0
; CHECK-NEXT:    ret
entry:
  %vcvth_n_s64_f16 = tail call i64 @llvm.aarch64.neon.vcvtfp2fxs.i64.f16(half %a, i32 1)
  ret i64 %vcvth_n_s64_f16
}

define dso_local i64 @test_vcvth_n_s64_f16_32(half %a) {
; CHECK-LABEL: test_vcvth_n_s64_f16_32:
; CHECK:         fcvtzs h0, h0, #32
; CHECK-NEXT:    fmov x0, d0
; CHECK-NEXT:    ret
entry:
  %vcvth_n_s64_f16 = tail call i64 @llvm.aarch64.neon.vcvtfp2fxs.i64.f16(half %a, i32 32)
  ret i64 %vcvth_n_s64_f16
}

define dso_local half @test_vcvth_n_f16_u16_1(i16 %a) {
; CHECK-LABEL: test_vcvth_n_f16_u16_1:
; CHECK:         ucvtf h0, h0, #1
; CHECK-NEXT:    ret
entry:
  %0 = zext i16 %a to i32
  %fcvth_n = tail call half @llvm.aarch64.neon.vcvtfxu2fp.f16.i32(i32 %0, i32 1)
  ret half %fcvth_n
}

define dso_local half @test_vcvth_n_f16_u16_16(i16 %a) {
; CHECK-LABEL: test_vcvth_n_f16_u16_16:
; CHECK:         ucvtf h0, h0, #16
; CHECK-NEXT:    ret
entry:
  %0 = zext i16 %a to i32
  %fcvth_n = tail call half @llvm.aarch64.neon.vcvtfxu2fp.f16.i32(i32 %0, i32 16)
  ret half %fcvth_n
}

define dso_local half @test_vcvth_n_f16_u32_1(i32 %a) {
; CHECK-LABEL: test_vcvth_n_f16_u32_1:
; CHECK:         fmov s0, w0
; CHECK-NEXT:    ucvtf h0, h0, #1
; CHECK-NEXT:    ret
entry:
  %vcvth_n_f16_u32 = tail call half @llvm.aarch64.neon.vcvtfxu2fp.f16.i32(i32 %a, i32 1)
  ret half %vcvth_n_f16_u32
}

define dso_local half @test_vcvth_n_f16_u32_16(i32 %a) {
; CHECK-LABEL: test_vcvth_n_f16_u32_16:
; CHECK:         ucvtf h0, h0, #16
; CHECK-NEXT:    ret
entry:
  %vcvth_n_f16_u32 = tail call half @llvm.aarch64.neon.vcvtfxu2fp.f16.i32(i32 %a, i32 16)
  ret half %vcvth_n_f16_u32
}

define dso_local i16 @test_vcvth_n_u16_f16_1(half %a) {
; CHECK-LABEL: test_vcvth_n_u16_f16_1:
; CHECK:         fcvtzu h0, h0, #1
; CHECK-NEXT:    fmov w0, s0
; CHECK-NEXT:    ret
entry:
  %fcvth_n = tail call i32 @llvm.aarch64.neon.vcvtfp2fxu.i32.f16(half %a, i32 1)
  %0 = trunc i32 %fcvth_n to i16
  ret i16 %0
}

define dso_local i16 @test_vcvth_n_u16_f16_16(half %a) {
; CHECK-LABEL: test_vcvth_n_u16_f16_16:
; CHECK:         fcvtzu h0, h0, #16
; CHECK-NEXT:    fmov w0, s0
; CHECK-NEXT:    ret
entry:
  %fcvth_n = tail call i32 @llvm.aarch64.neon.vcvtfp2fxu.i32.f16(half %a, i32 16)
  %0 = trunc i32 %fcvth_n to i16
  ret i16 %0
}

define dso_local i32 @test_vcvth_n_u32_f16_1(half %a) {
; CHECK-LABEL: test_vcvth_n_u32_f16_1:
; CHECK:         fcvtzu h0, h0, #1
; CHECK-NEXT:    fmov w0, s0
; CHECK-NEXT:    ret
entry:
  %vcvth_n_u32_f16 = tail call i32 @llvm.aarch64.neon.vcvtfp2fxu.i32.f16(half %a, i32 1)
  ret i32 %vcvth_n_u32_f16
}

define dso_local i32 @test_vcvth_n_u32_f16_16(half %a) {
; CHECK-LABEL: test_vcvth_n_u32_f16_16:
; CHECK:         fcvtzu h0, h0, #16
; CHECK-NEXT:    fmov w0, s0
; CHECK-NEXT:    ret
entry:
  %vcvth_n_u32_f16 = tail call i32 @llvm.aarch64.neon.vcvtfp2fxu.i32.f16(half %a, i32 16)
  ret i32 %vcvth_n_u32_f16
}
