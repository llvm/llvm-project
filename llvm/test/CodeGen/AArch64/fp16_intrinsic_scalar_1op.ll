; RUN: llc < %s -mtriple=aarch64-eabi -mattr=+v8.2a,+fullfp16  | FileCheck %s

declare i64 @llvm.aarch64.neon.fcvtpu.i64.f16(half)
declare i32 @llvm.aarch64.neon.fcvtpu.i32.f16(half)
declare i64 @llvm.aarch64.neon.fcvtps.i64.f16(half)
declare i32 @llvm.aarch64.neon.fcvtps.i32.f16(half)
declare i64 @llvm.aarch64.neon.fcvtnu.i64.f16(half)
declare i32 @llvm.aarch64.neon.fcvtnu.i32.f16(half)
declare i64 @llvm.aarch64.neon.fcvtns.i64.f16(half)
declare i32 @llvm.aarch64.neon.fcvtns.i32.f16(half)
declare i64 @llvm.aarch64.neon.fcvtmu.i64.f16(half)
declare i32 @llvm.aarch64.neon.fcvtmu.i32.f16(half)
declare i64 @llvm.aarch64.neon.fcvtms.i64.f16(half)
declare i32 @llvm.aarch64.neon.fcvtms.i32.f16(half)
declare i64 @llvm.aarch64.neon.fcvtau.i64.f16(half)
declare i32 @llvm.aarch64.neon.fcvtau.i32.f16(half)
declare i64 @llvm.aarch64.neon.fcvtas.i64.f16(half)
declare i32 @llvm.aarch64.neon.fcvtas.i32.f16(half)
declare half @llvm.aarch64.neon.frsqrte.f16(half)
declare half @llvm.aarch64.neon.frecpx.f16(half)
declare half @llvm.aarch64.neon.frecpe.f16(half)

define dso_local i16 @t2(half %a) {
; CHECK-LABEL: t2:
; CHECK:         fcmp h0, #0.0
; CHECK-NEXT:    csetm w0, eq
; CHECK-NEXT:    ret
entry:
  %0 = fcmp oeq half %a, 0xH0000
  %vceqz = sext i1 %0 to i16
  ret i16 %vceqz
}

define dso_local i16 @t3(half %a) {
; CHECK-LABEL: t3:
; CHECK:         fcmp h0, #0.0
; CHECK-NEXT:    csetm w0, ge
; CHECK-NEXT:    ret
entry:
  %0 = fcmp oge half %a, 0xH0000
  %vcgez = sext i1 %0 to i16
  ret i16 %vcgez
}

define dso_local i16 @t4(half %a) {
; CHECK-LABEL: t4:
; CHECK:         fcmp h0, #0.0
; CHECK-NEXT:    csetm w0, gt
; CHECK-NEXT:    ret
entry:
  %0 = fcmp ogt half %a, 0xH0000
  %vcgtz = sext i1 %0 to i16
  ret i16 %vcgtz
}

define dso_local i16 @t5(half %a) {
; CHECK-LABEL: t5:
; CHECK:         fcmp h0, #0.0
; CHECK-NEXT:    csetm w0, ls
; CHECK-NEXT:    ret
entry:
  %0 = fcmp ole half %a, 0xH0000
  %vclez = sext i1 %0 to i16
  ret i16 %vclez
}

define dso_local i16 @t6(half %a) {
; CHECK-LABEL: t6:
; CHECK:         fcmp h0, #0.0
; CHECK-NEXT:    csetm w0, mi
; CHECK-NEXT:    ret
entry:
  %0 = fcmp olt half %a, 0xH0000
  %vcltz = sext i1 %0 to i16
  ret i16 %vcltz
}

define dso_local half @t8(i32 %a) {
; CHECK-LABEL: t8:
; CHECK:         scvtf h0, w0
; CHECK-NEXT:    ret
entry:
  %0 = sitofp i32 %a to half
  ret half %0
}

define dso_local half @t9(i64 %a) {
; CHECK-LABEL: t9:
; CHECK:         scvtf h0, x0
; CHECK-NEXT:    ret
entry:
  %0 = sitofp i64 %a to half
  ret half %0
}

define dso_local half @t12(i64 %a) {
; CHECK-LABEL: t12:
; CHECK:         ucvtf h0, x0
; CHECK-NEXT:    ret
entry:
  %0 = uitofp i64 %a to half
  ret half %0
}

define dso_local i16 @t13(half %a) {
; CHECK-LABEL: t13:
; CHECK:         fcvtzs w0, h0
; CHECK-NEXT:    ret
entry:
  %0 = fptosi half %a to i16
  ret i16 %0
}

define dso_local i64 @t15(half %a) {
; CHECK-LABEL: t15:
; CHECK:         fcvtzs x0, h0
; CHECK-NEXT:    ret
entry:
  %0 = fptosi half %a to i64
  ret i64 %0
}

define dso_local i16 @t16(half %a) {
; CHECK-LABEL: t16:
; CHECK:         fcvtzs w0, h0
; CHECK-NEXT:    ret
entry:
  %0 = fptoui half %a to i16
  ret i16 %0
}

define dso_local i64 @t18(half %a) {
; CHECK-LABEL: t18:
; CHECK:         fcvtzu x0, h0
; CHECK-NEXT:    ret
entry:
  %0 = fptoui half %a to i64
  ret i64 %0
}

define dso_local i16 @t19(half %a) {
; CHECK-LABEL: t19:
; CHECK:         fcvtas w0, h0
; CHECK-NEXT:    ret
entry:
  %fcvt = tail call i32 @llvm.aarch64.neon.fcvtas.i32.f16(half %a)
  %0 = trunc i32 %fcvt to i16
  ret i16 %0
}

define dso_local i64 @t21(half %a) {
; CHECK-LABEL: t21:
; CHECK:         fcvtas x0, h0
; CHECK-NEXT:    ret
entry:
  %vcvtah_s64_f16 = tail call i64 @llvm.aarch64.neon.fcvtas.i64.f16(half %a)
  ret i64 %vcvtah_s64_f16
}

define dso_local i16 @t22(half %a) {
; CHECK-LABEL: t22:
; CHECK:         fcvtau w0, h0
; CHECK-NEXT:    ret
entry:
  %fcvt = tail call i32 @llvm.aarch64.neon.fcvtau.i32.f16(half %a)
  %0 = trunc i32 %fcvt to i16
  ret i16 %0
}

define dso_local i64 @t24(half %a) {
; CHECK-LABEL: t24:
; CHECK:         fcvtau x0, h0
; CHECK-NEXT:    ret
entry:
  %vcvtah_u64_f16 = tail call i64 @llvm.aarch64.neon.fcvtau.i64.f16(half %a)
  ret i64 %vcvtah_u64_f16
}

define dso_local i16 @t25(half %a) {
; CHECK-LABEL: t25:
; CHECK:         fcvtms w0, h0
; CHECK-NEXT:    ret
entry:
  %fcvt = tail call i32 @llvm.aarch64.neon.fcvtms.i32.f16(half %a)
  %0 = trunc i32 %fcvt to i16
  ret i16 %0
}

define dso_local i64 @t27(half %a) {
; CHECK-LABEL: t27:
; CHECK:         fcvtms x0, h0
; CHECK-NEXT:    ret
entry:
  %vcvtmh_s64_f16 = tail call i64 @llvm.aarch64.neon.fcvtms.i64.f16(half %a)
  ret i64 %vcvtmh_s64_f16
}

define dso_local i16 @t28(half %a) {
; CHECK-LABEL: t28:
; CHECK:         fcvtmu w0, h0
; CHECK-NEXT:    ret
entry:
  %fcvt = tail call i32 @llvm.aarch64.neon.fcvtmu.i32.f16(half %a)
  %0 = trunc i32 %fcvt to i16
  ret i16 %0
}

define dso_local i64 @t30(half %a) {
; CHECK-LABEL: t30:
; CHECK:         fcvtmu x0, h0
; CHECK-NEXT:    ret
entry:
  %vcvtmh_u64_f16 = tail call i64 @llvm.aarch64.neon.fcvtmu.i64.f16(half %a)
  ret i64 %vcvtmh_u64_f16
}

define dso_local i16 @t31(half %a) {
; CHECK-LABEL: t31:
; CHECK:         fcvtns w0, h0
; CHECK-NEXT:    ret
entry:
  %fcvt = tail call i32 @llvm.aarch64.neon.fcvtns.i32.f16(half %a)
  %0 = trunc i32 %fcvt to i16
  ret i16 %0
}

define dso_local i64 @t33(half %a) {
; CHECK-LABEL: t33:
; CHECK:         fcvtns x0, h0
; CHECK-NEXT:    ret
entry:
  %vcvtnh_s64_f16 = tail call i64 @llvm.aarch64.neon.fcvtns.i64.f16(half %a)
  ret i64 %vcvtnh_s64_f16
}

define dso_local i16 @t34(half %a) {
; CHECK-LABEL: t34:
; CHECK:         fcvtnu w0, h0
; CHECK-NEXT:    ret
entry:
  %fcvt = tail call i32 @llvm.aarch64.neon.fcvtnu.i32.f16(half %a)
  %0 = trunc i32 %fcvt to i16
  ret i16 %0
}

define dso_local i64 @t36(half %a) {
; CHECK-LABEL: t36:
; CHECK:         fcvtnu x0, h0
; CHECK-NEXT:    ret
entry:
  %vcvtnh_u64_f16 = tail call i64 @llvm.aarch64.neon.fcvtnu.i64.f16(half %a)
  ret i64 %vcvtnh_u64_f16
}

define dso_local i16 @t37(half %a) {
; CHECK-LABEL: t37:
; CHECK:         fcvtps w0, h0
; CHECK-NEXT:    ret
entry:
  %fcvt = tail call i32 @llvm.aarch64.neon.fcvtps.i32.f16(half %a)
  %0 = trunc i32 %fcvt to i16
  ret i16 %0
}

define dso_local i64 @t39(half %a) {
; CHECK-LABEL: t39:
; CHECK:         fcvtps x0, h0
; CHECK-NEXT:    ret
entry:
  %vcvtph_s64_f16 = tail call i64 @llvm.aarch64.neon.fcvtps.i64.f16(half %a)
  ret i64 %vcvtph_s64_f16
}

define dso_local i16 @t40(half %a) {
; CHECK-LABEL: t40:
; CHECK:         fcvtpu w0, h0
; CHECK-NEXT:    ret
entry:
  %fcvt = tail call i32 @llvm.aarch64.neon.fcvtpu.i32.f16(half %a)
  %0 = trunc i32 %fcvt to i16
  ret i16 %0
}

define dso_local i64 @t42(half %a) {
; CHECK-LABEL: t42:
; CHECK:         fcvtpu x0, h0
; CHECK-NEXT:    ret
entry:
  %vcvtph_u64_f16 = tail call i64 @llvm.aarch64.neon.fcvtpu.i64.f16(half %a)
  ret i64 %vcvtph_u64_f16
}

define dso_local half @t44(half %a) {
; CHECK-LABEL: t44:
; CHECK:         frecpe h0, h0
; CHECK-NEXT:    ret
entry:
  %vrecpeh_f16 = tail call half @llvm.aarch64.neon.frecpe.f16(half %a)
  ret half %vrecpeh_f16
}

define dso_local half @t45(half %a) {
; CHECK-LABEL: t45:
; CHECK:         frecpx h0, h0
; CHECK-NEXT:    ret
entry:
  %vrecpxh_f16 = tail call half @llvm.aarch64.neon.frecpx.f16(half %a)
  ret half %vrecpxh_f16
}

define dso_local half @t53(half %a) {
; CHECK-LABEL: t53:
; CHECK:         frsqrte h0, h0
; CHECK-NEXT:    ret
entry:
  %vrsqrteh_f16 = tail call half @llvm.aarch64.neon.frsqrte.f16(half %a)
  ret half %vrsqrteh_f16
}
