; RUN: opt -early-cse -S < %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

; CHECK-LABEL: define <4 x i32*> @fixed_length_version_first() {
; CHECK-NEXT:  ret <4 x i32*> undef
define <4 x i32*> @fixed_length_version_first() {
  %ptr = getelementptr i32, <4 x i32*> undef, <4 x i64> undef
  ret <4 x i32*> %ptr
}

; CHECK-LABEL: define <4 x <4 x i32>*> @fixed_length_version_second() {
; CHECK-NEXT:  ret <4 x <4 x i32>*> undef
define <4 x <4 x i32>*> @fixed_length_version_second() {
  %ptr = getelementptr <4 x i32>, <4 x i32>* undef, <4 x i64> undef
  ret <4 x <4 x i32>*> %ptr
}

; CHECK-LABEL: define <vscale x 4 x i32*> @vscale_version_first() {
; CHECK-NEXT:  ret <vscale x 4 x i32*> undef
define <vscale x 4 x i32*> @vscale_version_first() {
  %ptr = getelementptr i32, <vscale x 4 x i32*> undef, <vscale x 4 x i64> undef
  ret <vscale x 4 x i32*> %ptr
}

; CHECK-LABEL: define <vscale x 4 x <vscale x 4 x i32>*> @vscale_version_second() {
; CHECK-NEXT:  ret <vscale x 4 x <vscale x 4 x i32>*> undef
define <vscale x 4 x <vscale x 4 x i32>*> @vscale_version_second() {
  %ptr = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* undef, <vscale x 4 x i64> undef
  ret <vscale x 4 x <vscale x 4 x i32>*> %ptr
}
