; RUN: llc -ppc-scalar-mass -O3 -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -ppc-scalar-mass -O3 -mtriple=powerpc-ibm-aix-xcoff < %s | FileCheck %s
declare double @atan2 (double);

define double @atan2_bad(double %a) {
; CHECK-LABEL: atan2_bad
; CHECK-NOT: __xl_atan2_finite
; CHECK: blr
entry:
  %call = tail call nnan ninf afn nsz double @atan2(double %a)
  ret double %call
}