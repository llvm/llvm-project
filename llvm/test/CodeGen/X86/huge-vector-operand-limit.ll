; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512bw -O0 | FileCheck %s

; Vectors with more than SDNode::getMaxNumOperands() elements (#214666). Only two
; lanes are used because widened loads and stores of this size combine slowly.

define i8 @widen_to_65536(ptr %p) {
; CHECK-LABEL: widen_to_65536:
; CHECK: retq
  %w = load <65536 x i8>, ptr %p, align 16
  %v = call <65535 x i8> @llvm.vector.extract.v65535i8.v65536i8(<65536 x i8> %w, i64 0)
  %s = shl <65535 x i8> %v, splat (i8 5)
  %a = extractelement <65535 x i8> %s, i32 0
  %b = extractelement <65535 x i8> %s, i32 65534
  %r = add i8 %a, %b
  ret i8 %r
}

define void @exact_65536(ptr %p) {
; CHECK-LABEL: exact_65536:
; CHECK: retq
  %v = load <65536 x i8>, ptr %p, align 16
  %s = shl <65536 x i8> %v, splat (i8 5)
  store <65536 x i8> %s, ptr %p, align 16
  ret void
}

define i8 @widen_to_131072(ptr %p) {
; CHECK-LABEL: widen_to_131072:
; CHECK: retq
  %w = load <131072 x i8>, ptr %p, align 16
  %v = call <65537 x i8> @llvm.vector.extract.v65537i8.v131072i8(<131072 x i8> %w, i64 0)
  %s = shl <65537 x i8> %v, splat (i8 5)
  %a = extractelement <65537 x i8> %s, i32 0
  %b = extractelement <65537 x i8> %s, i32 65536
  %r = add i8 %a, %b
  ret i8 %r
}

define void @exact_131072(ptr %p) {
; CHECK-LABEL: exact_131072:
; CHECK: retq
  %v = load <131072 x i8>, ptr %p, align 16
  %s = shl <131072 x i8> %v, splat (i8 5)
  store <131072 x i8> %s, ptr %p, align 16
  ret void
}

define i8 @scalarized_arg_65537(<65537 x i8> %v) {
; CHECK-LABEL: scalarized_arg_65537:
; CHECK: retq
  %e = extractelement <65537 x i8> %v, i32 65536
  ret i8 %e
}

define i32 @pr214666(ptr %x) {
; CHECK-LABEL: pr214666:
; CHECK: retq
  %w = load <65536 x i32>, ptr %x, align 16
  %v = call <65535 x i32> @llvm.vector.extract.v65535i32.v65536i32(<65536 x i32> %w, i64 0)
  %s1 = shl <65535 x i32> %v, splat (i32 5)
  %s2 = shl <65535 x i32> %s1, zeroinitializer
  %a = extractelement <65535 x i32> %s2, i32 0
  %b = extractelement <65535 x i32> %s2, i32 65534
  %r = add i32 %a, %b
  ret i32 %r
}

declare <65535 x i8> @llvm.vector.extract.v65535i8.v65536i8(<65536 x i8>, i64)
declare <65537 x i8> @llvm.vector.extract.v65537i8.v131072i8(<131072 x i8>, i64)
declare <65535 x i32> @llvm.vector.extract.v65535i32.v65536i32(<65536 x i32>, i64)
