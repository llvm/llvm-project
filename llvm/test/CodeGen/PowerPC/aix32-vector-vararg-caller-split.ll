; RUN: not --crash llc -verify-machineinstrs -stop-before=ppc-vsx-copy -vec-extabi \
; RUN:     -mcpu=pwr7  -mtriple powerpc-ibm-aix-xcoff < %s 2>&1 | \
; RUN: FileCheck %s

define void @caller() {
entry:
  %call = tail call <4 x i32> (double, double, double, ...) @split_spill(double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, <4 x i32> <i32 1, i32 2, i32 3, i32 4>)
  ret void
}

declare <4 x i32> @split_spill(double, double, double, ...)

; CHECK: ERROR: Unexpected register handling for calling convention.
