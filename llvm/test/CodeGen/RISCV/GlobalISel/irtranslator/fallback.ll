; RUN: llc -mtriple=riscv64 -mattr='+v' -O0 -global-isel -global-isel-abort=2 -pass-remarks-missed='gisel*' -verify-machineinstrs %s -o %t.out 2> %t.err
; RUN: FileCheck %s --check-prefix=FALLBACK-WITH-REPORT-OUT < %t.out
; RUN: FileCheck %s --check-prefix=FALLBACK-WITH-REPORT-ERR < %t.err


declare <vscale x 1 x i8> @llvm.riscv.vadd.nxv1i8.nxv1i8(
  <vscale x 1 x i8>,
  <vscale x 1 x i8>,
  <vscale x 1 x i8>,
  i64)

; FALLBACK_WITH_REPORT_ERR:  <unknown>:0:0: unable to translate instruction: call:
; FALLBACK-WITH-REPORT-OUT-LABEL: scalable_arg
define <vscale x 1 x i8> @scalable_arg(<vscale x 1 x i8> %0, <vscale x 1 x i8> %1, i64 %2) nounwind {
entry:
  %a = call <vscale x 1 x i8> @llvm.riscv.vadd.nxv1i8.nxv1i8(
    <vscale x 1 x i8> undef,
    <vscale x 1 x i8> %0,
    <vscale x 1 x i8> %1,
    i64 %2)

  ret <vscale x 1 x i8> %a
}

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to translate instruction: call:
; FALLBACK-WITH-REPORT-OUT-LABEL: scalable_inst
define <vscale x 1 x i8> @scalable_inst(i64 %0) nounwind {
entry:
  %a = call <vscale x 1 x i8> @llvm.riscv.vadd.nxv1i8.nxv1i8(
    <vscale x 1 x i8> undef,
    <vscale x 1 x i8> undef,
    <vscale x 1 x i8> undef,
    i64 %0)

  ret <vscale x 1 x i8> %a
}

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to translate instruction: alloca:
; FALLBACK-WITH-REPORT-OUT-LABEL: scalable_alloca
define void @scalable_alloca() #1 {
  %local0 = alloca <vscale x 16 x i8>
  load volatile <vscale x 16 x i8>, ptr %local0
  ret void
}
