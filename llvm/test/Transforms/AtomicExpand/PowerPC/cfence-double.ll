; RUN: not --crash opt -S -atomic-expand -mtriple=powerpc64le-unknown-unknown \
; RUN:   -opaque-pointers < %s 2>&1 | FileCheck %s
; RUN: not --crash opt -S -atomic-expand -mtriple=powerpc64-unknown-unknown \
; RUN:   -opaque-pointers < %s 2>&1 | FileCheck %s

; CHECK: Intrinsic has incorrect argument type!
; CHECK: ptr @llvm.ppc.cfence.f64
define double @foo(double* %dp) {
entry:
  %0 = load atomic double, double* %dp acquire, align 8
  ret double %0
}
