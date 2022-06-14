; RUN: not --crash opt -S -atomic-expand -mtriple=powerpc64le-unknown-unknown \
; RUN:   -opaque-pointers < %s 2>&1 | FileCheck %s
; RUN: not --crash opt -S -atomic-expand -mtriple=powerpc64-unknown-unknown \
; RUN:   -opaque-pointers < %s 2>&1 | FileCheck %s

; CHECK: Intrinsic has incorrect argument type!
; CHECK: ptr @llvm.ppc.cfence.f32
define float @bar(float* %fp) {
entry:
  %0 = load atomic float, float* %fp acquire, align 4
  ret float %0
}
