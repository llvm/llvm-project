; RUN: llc < %s -mtriple=aarch64-unknown-linux-gnu -O0 -global-isel=0 | FileCheck %s

; Vectors with more than 65535 elements (#214666). Only two lanes are used
; because widened loads and stores of this size combine slowly.

define i8 @widen_to_65536(ptr %p) {
; CHECK-LABEL: widen_to_65536:
; CHECK: ret
  %w = load <65536 x i8>, ptr %p, align 16
  %v = call <65535 x i8> @llvm.vector.extract.v65535i8.v65536i8(<65536 x i8> %w, i64 0)
  %s = shl <65535 x i8> %v, splat (i8 5)
  %a = extractelement <65535 x i8> %s, i32 0
  %b = extractelement <65535 x i8> %s, i32 65534
  %r = add i8 %a, %b
  ret i8 %r
}

define i8 @widen_to_131072(ptr %p) {
; CHECK-LABEL: widen_to_131072:
; CHECK: ret
  %w = load <131072 x i8>, ptr %p, align 16
  %v = call <65537 x i8> @llvm.vector.extract.v65537i8.v131072i8(<131072 x i8> %w, i64 0)
  %s = shl <65537 x i8> %v, splat (i8 5)
  %a = extractelement <65537 x i8> %s, i32 0
  %b = extractelement <65537 x i8> %s, i32 65536
  %r = add i8 %a, %b
  ret i8 %r
}

declare <65535 x i8> @llvm.vector.extract.v65535i8.v65536i8(<65536 x i8>, i64)
declare <65537 x i8> @llvm.vector.extract.v65537i8.v131072i8(<131072 x i8>, i64)
