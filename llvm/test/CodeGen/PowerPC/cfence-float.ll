; RUN: not --crash llc -opaque-pointers -mtriple=powerpc64le-unknown-unknown \
; RUN:   < %s 2>&1 | FileCheck %s
; RUN: not --crash llc -opaque-pointers -mtriple=powerpc64-unknown-unknown \
; RUN:   < %s 2>&1 | FileCheck %s

; CHECK: Intrinsic has incorrect argument type!
; CHECK: ptr @llvm.ppc.cfence.f32
define void @bar() {
entry:
  %0 = load atomic float, float* undef acquire, align 8
  ret void
}
