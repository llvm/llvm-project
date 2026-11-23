; RUN: llc -mtriple=riscv32 -mattr=+zfh -verify-machineinstrs -target-abi=ilp32f < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK-F16,CHECK-ZFH
; RUN: llc -mtriple=riscv64 -mattr=+zfh -verify-machineinstrs -target-abi=lp64f < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK-F16,CHECK-ZFH
; RUN: llc -mtriple=riscv32 -mattr=+zhinx -verify-machineinstrs -target-abi=ilp32 < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK-F16,CHECK-ZHINX
; RUN: llc -mtriple=riscv64 -mattr=+zhinx -verify-machineinstrs -target-abi=lp64 < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK-F16,CHECK-ZHINX

define half @floor_f16_optsize(half %a) nounwind optsize {
; CHECK-ZFH-LABEL: floor_f16_optsize:
; CHECK-ZFH:       # %bb.0:
; CHECK-ZFH-NEXT:    li a0, 25
; CHECK-ZFH-NEXT:    slli a0, a0, 10
; CHECK-ZFH-NEXT:    fmv.h.x fa5, a0
; CHECK-ZFH-NEXT:    fabs.h fa4, fa0
; CHECK-ZFH-NEXT:    flt.h a0, fa4, fa5
; CHECK-ZFH-NEXT:    beqz a0, .LBB0_2
; CHECK-ZFH-NEXT:  # %bb.1:
; CHECK-ZFH-NEXT:    fcvt.w.h a0, fa0, rdn
; CHECK-ZFH-NEXT:    fcvt.h.w fa5, a0, rdn
; CHECK-ZFH-NEXT:    fsgnj.h fa0, fa5, fa0
; CHECK-ZFH-NEXT:  .LBB0_2:
; CHECK-ZFH-NEXT:    ret
;
; CHECK-ZHINX-LABEL: floor_f16_optsize:
; CHECK-ZHINX:       # %bb.0:
; CHECK-ZHINX-NEXT:    fabs.h a1, a0
; CHECK-ZHINX-NEXT:    li a2, 25
; CHECK-ZHINX-NEXT:    slli a2, a2, 10
; CHECK-ZHINX-NEXT:    flt.h a1, a1, a2
; CHECK-ZHINX-NEXT:    beqz a1, .LBB0_2
; CHECK-ZHINX-NEXT:  # %bb.1:
; CHECK-ZHINX-NEXT:    fcvt.w.h a1, a0, rdn
; CHECK-ZHINX-NEXT:    fcvt.h.w a1, a1, rdn
; CHECK-ZHINX-NEXT:    fsgnj.h a0, a1, a0
; CHECK-ZHINX-NEXT:  .LBB0_2:
; CHECK-ZHINX-NEXT:    ret
  %1 = call half @llvm.floor.f16(half %a)
  ret half %1
}

define half @ceil_f16_optsize(half %a) nounwind optsize {
; CHECK-ZFH-LABEL: ceil_f16_optsize:
; CHECK-ZFH:       # %bb.0:
; CHECK-ZFH-NEXT:    li a0, 25
; CHECK-ZFH-NEXT:    slli a0, a0, 10
; CHECK-ZFH-NEXT:    fmv.h.x fa5, a0
; CHECK-ZFH-NEXT:    fabs.h fa4, fa0
; CHECK-ZFH-NEXT:    flt.h a0, fa4, fa5
; CHECK-ZFH-NEXT:    beqz a0, .LBB1_2
; CHECK-ZFH-NEXT:  # %bb.1:
; CHECK-ZFH-NEXT:    fcvt.w.h a0, fa0, rup
; CHECK-ZFH-NEXT:    fcvt.h.w fa5, a0, rup
; CHECK-ZFH-NEXT:    fsgnj.h fa0, fa5, fa0
; CHECK-ZFH-NEXT:  .LBB1_2:
; CHECK-ZFH-NEXT:    ret
;
; CHECK-ZHINX-LABEL: ceil_f16_optsize:
; CHECK-ZHINX:       # %bb.0:
; CHECK-ZHINX-NEXT:    fabs.h a1, a0
; CHECK-ZHINX-NEXT:    li a2, 25
; CHECK-ZHINX-NEXT:    slli a2, a2, 10
; CHECK-ZHINX-NEXT:    flt.h a1, a1, a2
; CHECK-ZHINX-NEXT:    beqz a1, .LBB1_2
; CHECK-ZHINX-NEXT:  # %bb.1:
; CHECK-ZHINX-NEXT:    fcvt.w.h a1, a0, rup
; CHECK-ZHINX-NEXT:    fcvt.h.w a1, a1, rup
; CHECK-ZHINX-NEXT:    fsgnj.h a0, a1, a0
; CHECK-ZHINX-NEXT:  .LBB1_2:
; CHECK-ZHINX-NEXT:    ret
  %1 = call half @llvm.ceil.f16(half %a)
  ret half %1
}

define half @trunc_f16_optsize(half %a) nounwind optsize {
; CHECK-ZFH-LABEL: trunc_f16_optsize:
; CHECK-ZFH:       # %bb.0:
; CHECK-ZFH-NEXT:    li a0, 25
; CHECK-ZFH-NEXT:    slli a0, a0, 10
; CHECK-ZFH-NEXT:    fmv.h.x fa5, a0
; CHECK-ZFH-NEXT:    fabs.h fa4, fa0
; CHECK-ZFH-NEXT:    flt.h a0, fa4, fa5
; CHECK-ZFH-NEXT:    beqz a0, .LBB2_2
; CHECK-ZFH-NEXT:  # %bb.1:
; CHECK-ZFH-NEXT:    fcvt.w.h a0, fa0, rtz
; CHECK-ZFH-NEXT:    fcvt.h.w fa5, a0, rtz
; CHECK-ZFH-NEXT:    fsgnj.h fa0, fa5, fa0
; CHECK-ZFH-NEXT:  .LBB2_2:
; CHECK-ZFH-NEXT:    ret
;
; CHECK-ZHINX-LABEL: trunc_f16_optsize:
; CHECK-ZHINX:       # %bb.0:
; CHECK-ZHINX-NEXT:    fabs.h a1, a0
; CHECK-ZHINX-NEXT:    li a2, 25
; CHECK-ZHINX-NEXT:    slli a2, a2, 10
; CHECK-ZHINX-NEXT:    flt.h a1, a1, a2
; CHECK-ZHINX-NEXT:    beqz a1, .LBB2_2
; CHECK-ZHINX-NEXT:  # %bb.1:
; CHECK-ZHINX-NEXT:    fcvt.w.h a1, a0, rtz
; CHECK-ZHINX-NEXT:    fcvt.h.w a1, a1, rtz
; CHECK-ZHINX-NEXT:    fsgnj.h a0, a1, a0
; CHECK-ZHINX-NEXT:  .LBB2_2:
; CHECK-ZHINX-NEXT:    ret
  %1 = call half @llvm.trunc.f16(half %a)
  ret half %1
}

define half @rint_f16_optsize(half %a) nounwind optsize {
; CHECK-ZFH-LABEL: rint_f16_optsize:
; CHECK-ZFH:       # %bb.0:
; CHECK-ZFH-NEXT:    li a0, 25
; CHECK-ZFH-NEXT:    slli a0, a0, 10
; CHECK-ZFH-NEXT:    fmv.h.x fa5, a0
; CHECK-ZFH-NEXT:    fabs.h fa4, fa0
; CHECK-ZFH-NEXT:    flt.h a0, fa4, fa5
; CHECK-ZFH-NEXT:    beqz a0, .LBB3_2
; CHECK-ZFH-NEXT:  # %bb.1:
; CHECK-ZFH-NEXT:    fcvt.w.h a0, fa0
; CHECK-ZFH-NEXT:    fcvt.h.w fa5, a0
; CHECK-ZFH-NEXT:    fsgnj.h fa0, fa5, fa0
; CHECK-ZFH-NEXT:  .LBB3_2:
; CHECK-ZFH-NEXT:    ret
;
; CHECK-ZHINX-LABEL: rint_f16_optsize:
; CHECK-ZHINX:       # %bb.0:
; CHECK-ZHINX-NEXT:    fabs.h a1, a0
; CHECK-ZHINX-NEXT:    li a2, 25
; CHECK-ZHINX-NEXT:    slli a2, a2, 10
; CHECK-ZHINX-NEXT:    flt.h a1, a1, a2
; CHECK-ZHINX-NEXT:    beqz a1, .LBB3_2
; CHECK-ZHINX-NEXT:  # %bb.1:
; CHECK-ZHINX-NEXT:    fcvt.w.h a1, a0
; CHECK-ZHINX-NEXT:    fcvt.h.w a1, a1
; CHECK-ZHINX-NEXT:    fsgnj.h a0, a1, a0
; CHECK-ZHINX-NEXT:  .LBB3_2:
; CHECK-ZHINX-NEXT:    ret
  %1 = call half @llvm.rint.f16(half %a)
  ret half %1
}

define half @round_f16_optsize(half %a) nounwind optsize {
; CHECK-ZFH-LABEL: round_f16_optsize:
; CHECK-ZFH:       # %bb.0:
; CHECK-ZFH-NEXT:    li a0, 25
; CHECK-ZFH-NEXT:    slli a0, a0, 10
; CHECK-ZFH-NEXT:    fmv.h.x fa5, a0
; CHECK-ZFH-NEXT:    fabs.h fa4, fa0
; CHECK-ZFH-NEXT:    flt.h a0, fa4, fa5
; CHECK-ZFH-NEXT:    beqz a0, .LBB4_2
; CHECK-ZFH-NEXT:  # %bb.1:
; CHECK-ZFH-NEXT:    fcvt.w.h a0, fa0, rmm
; CHECK-ZFH-NEXT:    fcvt.h.w fa5, a0, rmm
; CHECK-ZFH-NEXT:    fsgnj.h fa0, fa5, fa0
; CHECK-ZFH-NEXT:  .LBB4_2:
; CHECK-ZFH-NEXT:    ret
;
; CHECK-ZHINX-LABEL: round_f16_optsize:
; CHECK-ZHINX:       # %bb.0:
; CHECK-ZHINX-NEXT:    fabs.h a1, a0
; CHECK-ZHINX-NEXT:    li a2, 25
; CHECK-ZHINX-NEXT:    slli a2, a2, 10
; CHECK-ZHINX-NEXT:    flt.h a1, a1, a2
; CHECK-ZHINX-NEXT:    beqz a1, .LBB4_2
; CHECK-ZHINX-NEXT:  # %bb.1:
; CHECK-ZHINX-NEXT:    fcvt.w.h a1, a0, rmm
; CHECK-ZHINX-NEXT:    fcvt.h.w a1, a1, rmm
; CHECK-ZHINX-NEXT:    fsgnj.h a0, a1, a0
; CHECK-ZHINX-NEXT:  .LBB4_2:
; CHECK-ZHINX-NEXT:    ret
  %1 = call half @llvm.round.f16(half %a)
  ret half %1
}

define half @roundeven_f16_minsize(half %a) nounwind minsize {
; CHECK-ZFH-LABEL: roundeven_f16_minsize:
; CHECK-ZFH:       # %bb.0:
; CHECK-ZFH-NEXT:    li a0, 25
; CHECK-ZFH-NEXT:    slli a0, a0, 10
; CHECK-ZFH-NEXT:    fmv.h.x fa5, a0
; CHECK-ZFH-NEXT:    fabs.h fa4, fa0
; CHECK-ZFH-NEXT:    flt.h a0, fa4, fa5
; CHECK-ZFH-NEXT:    beqz a0, .LBB5_2
; CHECK-ZFH-NEXT:  # %bb.1:
; CHECK-ZFH-NEXT:    fcvt.w.h a0, fa0, rne
; CHECK-ZFH-NEXT:    fcvt.h.w fa5, a0, rne
; CHECK-ZFH-NEXT:    fsgnj.h fa0, fa5, fa0
; CHECK-ZFH-NEXT:  .LBB5_2:
; CHECK-ZFH-NEXT:    ret
;
; CHECK-ZHINX-LABEL: roundeven_f16_minsize:
; CHECK-ZHINX:       # %bb.0:
; CHECK-ZHINX-NEXT:    fabs.h a1, a0
; CHECK-ZHINX-NEXT:    li a2, 25
; CHECK-ZHINX-NEXT:    slli a2, a2, 10
; CHECK-ZHINX-NEXT:    flt.h a1, a1, a2
; CHECK-ZHINX-NEXT:    beqz a1, .LBB5_2
; CHECK-ZHINX-NEXT:  # %bb.1:
; CHECK-ZHINX-NEXT:    fcvt.w.h a1, a0, rne
; CHECK-ZHINX-NEXT:    fcvt.h.w a1, a1, rne
; CHECK-ZHINX-NEXT:    fsgnj.h a0, a1, a0
; CHECK-ZHINX-NEXT:  .LBB5_2:
; CHECK-ZHINX-NEXT:    ret
  %1 = call half @llvm.roundeven.f16(half %a)
  ret half %1
}

define float @floor_f32_optsize(float %a) nounwind optsize {
; CHECK-F16-LABEL: floor_f32_optsize:
; CHECK-F16:       # %bb.0:
; CHECK-F16-NEXT:    tail floorf
  %1 = call float @llvm.floor.f32(float %a)
  ret float %1
}

define float @ceil_f32_optsize(float %a) nounwind optsize {
; CHECK-F16-LABEL: ceil_f32_optsize:
; CHECK-F16:       # %bb.0:
; CHECK-F16-NEXT:    tail ceilf
  %1 = call float @llvm.ceil.f32(float %a)
  ret float %1
}
