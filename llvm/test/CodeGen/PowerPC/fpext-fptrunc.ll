; RUN: llc -O3 -mtriple=powerpc64-ibm-aix -verify-machineinstrs < %s | FileCheck %s
;
; Fast-math flags are propagated to fpext, which can enable folds on rounds
; that previously lacked it. We should avoid folding fpext(fptrunc(x,0)),
; essentially when the TRUNC flag is 1, as this is not deemed safe.
;
; CHECK-LABEL: test_fpext_fptrunc:
; CHECK:       frsp
; CHECK-NOT:   xsredp
define i1 @test_fpext_fptrunc(double %a, double %b, ppc_fp128 %stored) {
  %div   = fdiv fast double %a, %b
  %trunc = fptrunc fast double %div to float
  %ext   = fpext  fast float  %trunc to ppc_fp128
  %cmp   = fcmp fast une ppc_fp128 %stored, %ext
  ret i1 %cmp
}
