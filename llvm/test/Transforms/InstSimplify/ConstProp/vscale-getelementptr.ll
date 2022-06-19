; RUN: opt -early-cse -earlycse-debug-hash -S < %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

; CHECK-LABEL: define <4 x ptr> @fixed_length_version_first() {
; CHECK-NEXT:  ret <4 x ptr> undef
define <4 x ptr> @fixed_length_version_first() {
  %ptr = getelementptr i32, <4 x ptr> undef, <4 x i64> undef
  ret <4 x ptr> %ptr
}

; CHECK-LABEL: define <4 x ptr> @fixed_length_version_second() {
; CHECK-NEXT:  ret <4 x ptr> undef
define <4 x ptr> @fixed_length_version_second() {
  %ptr = getelementptr <4 x i32>, ptr undef, <4 x i64> undef
  ret <4 x ptr> %ptr
}

; CHECK-LABEL: define <vscale x 4 x ptr> @vscale_version_first() {
; CHECK-NEXT:  ret <vscale x 4 x ptr> undef
define <vscale x 4 x ptr> @vscale_version_first() {
  %ptr = getelementptr i32, <vscale x 4 x ptr> undef, <vscale x 4 x i64> undef
  ret <vscale x 4 x ptr> %ptr
}

; CHECK-LABEL: define <vscale x 4 x ptr> @vscale_version_second() {
; CHECK-NEXT:  ret <vscale x 4 x ptr> undef
define <vscale x 4 x ptr> @vscale_version_second() {
  %ptr = getelementptr <vscale x 4 x i32>, ptr undef, <vscale x 4 x i64> undef
  ret <vscale x 4 x ptr> %ptr
}
