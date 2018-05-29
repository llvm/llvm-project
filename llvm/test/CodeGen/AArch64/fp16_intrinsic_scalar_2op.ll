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
