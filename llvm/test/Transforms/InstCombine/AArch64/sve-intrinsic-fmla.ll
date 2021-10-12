; RUN: opt -S -instcombine < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

define dso_local <vscale x 8 x half> @combine_fmla(<vscale x 16 x i1> %0, <vscale x 8 x half> %1, <vscale x 8 x half> %2, <vscale x 8 x half> %3) local_unnamed_addr #0 {
; CHECK-LABEL: @combine_fmla
; CHECK-NEXT:  %5 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %0)
; CHECK-NEXT:  %6 = call fast <vscale x 8 x half> @llvm.aarch64.sve.fmla.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %1, <vscale x 8 x half> %2, <vscale x 8 x half> %3)
; CHECK-NEXT:  ret <vscale x 8 x half> %6
  %5 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %0)
  %6 = tail call fast <vscale x 8 x half> @llvm.aarch64.sve.fmul.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %2, <vscale x 8 x half> %3)
  %7 = tail call fast <vscale x 8 x half> @llvm.aarch64.sve.fadd.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %1, <vscale x 8 x half> %6)
  ret <vscale x 8 x half> %7
}

define dso_local <vscale x 8 x half> @neg_combine_fmla_contract_flag_only(<vscale x 16 x i1> %0, <vscale x 8 x half> %1, <vscale x 8 x half> %2, <vscale x 8 x half> %3) local_unnamed_addr #0 {
; CHECK-LABEL: @neg_combine_fmla_contract_flag_only
; CHECK-NEXT:  %5 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %0)
; CHECK-NEXT:  %6 = tail call contract <vscale x 8 x half> @llvm.aarch64.sve.fmul.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %2, <vscale x 8 x half> %3)
; CHECK-NEXT:  %7 = tail call contract <vscale x 8 x half> @llvm.aarch64.sve.fadd.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %1, <vscale x 8 x half> %6)
; CHECK-NEXT:  ret <vscale x 8 x half> %7
  %5 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %0)
  %6 = tail call contract <vscale x 8 x half> @llvm.aarch64.sve.fmul.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %2, <vscale x 8 x half> %3)
  %7 = tail call contract <vscale x 8 x half> @llvm.aarch64.sve.fadd.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %1, <vscale x 8 x half> %6)
  ret <vscale x 8 x half> %7
}

define dso_local <vscale x 8 x half> @neg_combine_fmla_reassoc_flag_only(<vscale x 16 x i1> %0, <vscale x 8 x half> %1, <vscale x 8 x half> %2, <vscale x 8 x half> %3) local_unnamed_addr #0 {
; CHECK-LABEL: @neg_combine_fmla_reassoc_flag_only
; CHECK-NEXT:  %5 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %0)
; CHECK-NEXT:  %6 = tail call reassoc <vscale x 8 x half> @llvm.aarch64.sve.fmul.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %2, <vscale x 8 x half> %3)
; CHECK-NEXT:  %7 = tail call reassoc <vscale x 8 x half> @llvm.aarch64.sve.fadd.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %1, <vscale x 8 x half> %6)
; CHECK-NEXT:  ret <vscale x 8 x half> %7
  %5 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %0)
  %6 = tail call reassoc <vscale x 8 x half> @llvm.aarch64.sve.fmul.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %2, <vscale x 8 x half> %3)
  %7 = tail call reassoc <vscale x 8 x half> @llvm.aarch64.sve.fadd.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %1, <vscale x 8 x half> %6)
  ret <vscale x 8 x half> %7
}

define dso_local <vscale x 8 x half> @neg_combine_fmla_min_flags(<vscale x 16 x i1> %0, <vscale x 8 x half> %1, <vscale x 8 x half> %2, <vscale x 8 x half> %3) local_unnamed_addr #0 {
; CHECK-LABEL: @neg_combine_fmla_min_flags
; CHECK-NEXT:  %5 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %0)
; CHECK-NEXT:  %6 = call reassoc contract <vscale x 8 x half> @llvm.aarch64.sve.fmla.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %1, <vscale x 8 x half> %2, <vscale x 8 x half> %3)
; CHECK-NEXT:  ret <vscale x 8 x half> %6
  %5 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %0)
  %6 = tail call reassoc contract <vscale x 8 x half> @llvm.aarch64.sve.fmul.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %2, <vscale x 8 x half> %3)
  %7 = tail call reassoc contract <vscale x 8 x half> @llvm.aarch64.sve.fadd.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %1, <vscale x 8 x half> %6)
  ret <vscale x 8 x half> %7
}

define dso_local <vscale x 8 x half> @neg_combine_fmla_no_fast_flag(<vscale x 16 x i1> %0, <vscale x 8 x half> %1, <vscale x 8 x half> %2, <vscale x 8 x half> %3) local_unnamed_addr #0 {
; CHECK-LABEL: @neg_combine_fmla_no_fast_flag
; CHECK-NEXT:  %5 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %0)
; CHECK-NEXT:  %6 = tail call <vscale x 8 x half> @llvm.aarch64.sve.fmul.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %2, <vscale x 8 x half> %3)
; CHECK-NEXT:  %7 = tail call <vscale x 8 x half> @llvm.aarch64.sve.fadd.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %1, <vscale x 8 x half> %6)
; CHECK-NEXT:  ret <vscale x 8 x half> %7
  %5 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %0)
  %6 = tail call <vscale x 8 x half> @llvm.aarch64.sve.fmul.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %2, <vscale x 8 x half> %3)
  %7 = tail call <vscale x 8 x half> @llvm.aarch64.sve.fadd.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %1, <vscale x 8 x half> %6)
  ret <vscale x 8 x half> %7
}

define dso_local <vscale x 8 x half> @neg_combine_fmla_no_fmul(<vscale x 16 x i1> %0, <vscale x 8 x half> %1, <vscale x 8 x half> %2, <vscale x 8 x half> %3) local_unnamed_addr #0 {
; CHECK-LABEL: @neg_combine_fmla_no_fmul
; CHECK-NEXT:  %5 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %0)
; CHECK-NEXT:  %6 = tail call fast <vscale x 8 x half> @llvm.aarch64.sve.fadd.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %2, <vscale x 8 x half> %3)
; CHECK-NEXT:  %7 = tail call fast <vscale x 8 x half> @llvm.aarch64.sve.fadd.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %1, <vscale x 8 x half> %6)
; CHECK-NEXT:  ret <vscale x 8 x half> %7
  %5 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %0)
  %6 = tail call fast <vscale x 8 x half> @llvm.aarch64.sve.fadd.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %2, <vscale x 8 x half> %3)
  %7 = tail call fast <vscale x 8 x half> @llvm.aarch64.sve.fadd.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %1, <vscale x 8 x half> %6)
  ret <vscale x 8 x half> %7
}

define dso_local <vscale x 8 x half> @neg_combine_fmla_neq_pred(<vscale x 16 x i1> %0, <vscale x 8 x half> %1, <vscale x 8 x half> %2, <vscale x 8 x half> %3) local_unnamed_addr #0 {
; CHECK-LABEL: @neg_combine_fmla_neq_pred
; CHECK-NEXT:  %5 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %0)
; CHECK-NEXT:  %6 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 5)
; CHECK-NEXT:  %7 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %6)
; CHECK-NEXT:  %8 = tail call fast <vscale x 8 x half> @llvm.aarch64.sve.fmul.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %2, <vscale x 8 x half> %3)
; CHECK-NEXT:  %9 = tail call fast <vscale x 8 x half> @llvm.aarch64.sve.fadd.nxv8f16(<vscale x 8 x i1> %7, <vscale x 8 x half> %1, <vscale x 8 x half> %8)
; ret <vscale x 8 x half> %9
  %5 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %0)
  %6 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 5)
  %7 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %6)
  %8 = tail call fast <vscale x 8 x half> @llvm.aarch64.sve.fmul.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %2, <vscale x 8 x half> %3)
  %9 = tail call fast <vscale x 8 x half> @llvm.aarch64.sve.fadd.nxv8f16(<vscale x 8 x i1> %7, <vscale x 8 x half> %1, <vscale x 8 x half> %8)
  ret <vscale x 8 x half> %9
}

define dso_local <vscale x 8 x half> @neg_combine_fmla_two_fmul_uses(<vscale x 16 x i1> %0, <vscale x 8 x half> %1, <vscale x 8 x half> %2, <vscale x 8 x half> %3) local_unnamed_addr #0 {
; CHECK-LABEL: @neg_combine_fmla_two_fmul_uses
; CHECK-NEXT:  %5 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %0)
; CHECK-NEXT:  %6 = tail call fast <vscale x 8 x half> @llvm.aarch64.sve.fmul.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %2, <vscale x 8 x half> %3)
; CHECK-NEXT:  %7 = tail call fast <vscale x 8 x half> @llvm.aarch64.sve.fadd.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %1, <vscale x 8 x half> %6)
; CHECK-NEXT:  %8 = tail call fast <vscale x 8 x half> @llvm.aarch64.sve.fadd.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %7, <vscale x 8 x half> %6)
; ret <vscale x 8 x half> %8
  %5 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %0)
  %6 = tail call fast <vscale x 8 x half> @llvm.aarch64.sve.fmul.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %2, <vscale x 8 x half> %3)
  %7 = tail call fast <vscale x 8 x half> @llvm.aarch64.sve.fadd.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %1, <vscale x 8 x half> %6)
  %8 = tail call fast <vscale x 8 x half> @llvm.aarch64.sve.fadd.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %7, <vscale x 8 x half> %6)
  ret <vscale x 8 x half> %8
}

define dso_local <vscale x 8 x half> @neg_combine_fmla_neq_flags(<vscale x 16 x i1> %0, <vscale x 8 x half> %1, <vscale x 8 x half> %2, <vscale x 8 x half> %3) local_unnamed_addr #0 {
; CHECK-LABEL: @neg_combine_fmla_neq_flags
; CHECK-NEXT:  %5 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %0)
; CHECK-NEXT:  %6 = tail call reassoc nnan contract <vscale x 8 x half> @llvm.aarch64.sve.fmul.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %2, <vscale x 8 x half> %3)
; CHECK-NEXT:  %7 = tail call reassoc contract <vscale x 8 x half> @llvm.aarch64.sve.fadd.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %1, <vscale x 8 x half> %6)
; ret <vscale x 8 x half> %7
  %5 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %0)
  %6 = tail call reassoc nnan contract <vscale x 8 x half> @llvm.aarch64.sve.fmul.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %2, <vscale x 8 x half> %3)
  %7 = tail call reassoc contract <vscale x 8 x half> @llvm.aarch64.sve.fadd.nxv8f16(<vscale x 8 x i1> %5, <vscale x 8 x half> %1, <vscale x 8 x half> %6)
  ret <vscale x 8 x half> %7
}

declare <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1>)
declare <vscale x 8 x half> @llvm.aarch64.sve.fmul.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 8 x half> @llvm.aarch64.sve.fadd.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32)
attributes #0 = { "target-features"="+sve" }
