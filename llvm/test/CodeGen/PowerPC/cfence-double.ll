; RUN: not --crash llc -opaque-pointers -mtriple=powerpc64le-unknown-unknown \
; RUN:   < %s 2>&1 | FileCheck %s
; RUN: not --crash llc -opaque-pointers -mtriple=powerpc64-unknown-unknown \
; RUN:   < %s 2>&1 | FileCheck %s

; CHECK: Intrinsic has incorrect argument type!
; CHECK: ptr @llvm.ppc.cfence.f64
define void @foo() {
entry:
  %0 = load atomic double, double* undef acquire, align 8
  ret void
}
