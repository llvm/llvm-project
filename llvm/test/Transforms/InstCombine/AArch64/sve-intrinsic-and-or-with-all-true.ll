; RUN: opt -passes=instcombine -mtriple aarch64 -mattr=+sve -S -o - < %s | FileCheck %s
;
; Test AArch64-specific InstCombine optimizations for SVE logical operations
; with all-true predicates.
; - a AND true = a
; - a OR true = true

declare <vscale x 16 x i1> @llvm.aarch64.sve.and.z.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.orr.z.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.pnext.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>)

define <vscale x 16 x i1> @test_sve_and_z_all_true_right(<vscale x 16 x i1> %a) {
; CHECK-LABEL: @test_sve_and_z_all_true_right(
; CHECK-NEXT:    ret <vscale x 16 x i1> [[A:%.*]]
  %all_true = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> splat (i1 true))
  %result = tail call <vscale x 16 x i1> @llvm.aarch64.sve.and.z.nxv16i1(<vscale x 16 x i1> splat (i1 true), <vscale x 16 x i1> %a, <vscale x 16 x i1> %all_true)
  ret <vscale x 16 x i1> %result
}

define <vscale x 16 x i1> @test_sve_and_z_all_true_left(<vscale x 16 x i1> %a) {
; CHECK-LABEL: @test_sve_and_z_all_true_left(
; CHECK-NEXT:    ret <vscale x 16 x i1> [[A:%.*]]
  %all_true = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> splat (i1 true))
  %result = tail call <vscale x 16 x i1> @llvm.aarch64.sve.and.z.nxv16i1(<vscale x 16 x i1> splat (i1 true), <vscale x 16 x i1> %all_true, <vscale x 16 x i1> %a)
  ret <vscale x 16 x i1> %result
}

define <vscale x 16 x i1> @test_sve_orr_z_all_true_right(<vscale x 16 x i1> %a) {
; CHECK-LABEL: @test_sve_orr_z_all_true_right(
; CHECK-NEXT:    [[ALL_TRUE:%.*]] = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> splat (i1 true))
; CHECK-NEXT:    ret <vscale x 16 x i1> [[ALL_TRUE]]
  %all_true = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> splat (i1 true))
  %result = tail call <vscale x 16 x i1> @llvm.aarch64.sve.orr.z.nxv16i1(<vscale x 16 x i1> splat (i1 true), <vscale x 16 x i1> %a, <vscale x 16 x i1> %all_true)
  ret <vscale x 16 x i1> %result
}

define <vscale x 16 x i1> @test_sve_orr_z_all_true_left(<vscale x 16 x i1> %a) {
; CHECK-LABEL: @test_sve_orr_z_all_true_left(
; CHECK-NEXT:    [[ALL_TRUE:%.*]] = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> splat (i1 true))
; CHECK-NEXT:    ret <vscale x 16 x i1> [[ALL_TRUE]]
  %all_true = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> splat (i1 true))
  %result = tail call <vscale x 16 x i1> @llvm.aarch64.sve.orr.z.nxv16i1(<vscale x 16 x i1> splat (i1 true), <vscale x 16 x i1> %all_true, <vscale x 16 x i1> %a)
  ret <vscale x 16 x i1> %result
}

define <vscale x 16 x i1> @test_original_bug_case(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %prev) {
; CHECK-LABEL: @test_original_bug_case(
; CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
; CHECK-NEXT:    [[TMP2:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PREV:%.*]])
; CHECK-NEXT:    [[TMP3:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.pnext.nxv8i1(<vscale x 8 x i1> [[TMP1]], <vscale x 8 x i1> [[TMP2]])
; CHECK-NEXT:    [[TMP4:%.*]] = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> [[TMP3]])
; CHECK-NEXT:    ret <vscale x 16 x i1> [[TMP4]]
  %1 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  %2 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %prev)
  %3 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.pnext.nxv8i1(<vscale x 8 x i1> %1, <vscale x 8 x i1> %2)
  %4 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %3)
  %5 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> splat (i1 true))
  %6 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.and.z.nxv16i1(<vscale x 16 x i1> splat (i1 true), <vscale x 16 x i1> %4, <vscale x 16 x i1> %5)
  ret <vscale x 16 x i1> %6
}

define <vscale x 16 x i1> @test_sve_and_z_not_all_true_predicate(<vscale x 16 x i1> %pred, <vscale x 16 x i1> %a) {
; CHECK-LABEL: @test_sve_and_z_not_all_true_predicate(
; CHECK-NEXT:    [[ALL_TRUE:%.*]] = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> splat (i1 true))
; CHECK-NEXT:    [[RESULT:%.*]] = tail call <vscale x 16 x i1> @llvm.aarch64.sve.and.z.nxv16i1(<vscale x 16 x i1> [[PRED:%.*]], <vscale x 16 x i1> [[A:%.*]], <vscale x 16 x i1> [[ALL_TRUE]])
; CHECK-NEXT:    ret <vscale x 16 x i1> [[RESULT]]
  %all_true = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> splat (i1 true))
  %result = tail call <vscale x 16 x i1> @llvm.aarch64.sve.and.z.nxv16i1(<vscale x 16 x i1> %pred, <vscale x 16 x i1> %a, <vscale x 16 x i1> %all_true)
  ret <vscale x 16 x i1> %result
}

define <vscale x 16 x i1> @test_sve_and_z_no_all_true_operands(<vscale x 16 x i1> %a, <vscale x 16 x i1> %b) {
; CHECK-LABEL: @test_sve_and_z_no_all_true_operands(
; CHECK-NEXT:    [[RESULT:%.*]] = tail call <vscale x 16 x i1> @llvm.aarch64.sve.and.z.nxv16i1(<vscale x 16 x i1> splat (i1 true), <vscale x 16 x i1> [[A:%.*]], <vscale x 16 x i1> [[B:%.*]])
; CHECK-NEXT:    ret <vscale x 16 x i1> [[RESULT]]
  %result = tail call <vscale x 16 x i1> @llvm.aarch64.sve.and.z.nxv16i1(<vscale x 16 x i1> splat (i1 true), <vscale x 16 x i1> %a, <vscale x 16 x i1> %b)
  ret <vscale x 16 x i1> %result
}
