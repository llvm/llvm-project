; RUN: opt -early-cse -earlycse-debug-hash -S < %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

; This test checks that SimplifyInstruction does not blow up in the face of
; a scalable shufflevector. vscale is a constant value known only at runtime.
; Therefore, it is not possible to know the concrete value of, or the length
; of the mask at compile time. Simplifications that depend on the value
; of the mask cannot be performed.

; Given the fact that the value of the mask is unknown at compile time for
; scalable vectors, very few simplifications will be done. Here, we want to
; see that the instruction can be passed to SimplifyInstruction and not crash
; the compiler. It happens to be the case that this will be the result.

; CHECK-LABEL: define <vscale x 8 x i1> @vscale_version()
; CHECK-NEXT: ret <vscale x 8 x i1> shufflevector (<vscale x 8 x i1> insertelement (<vscale x 8 x i1> undef, i1 true, i32 0), <vscale x 8 x i1> undef, <vscale x 8 x i32> zeroinitializer)

define <vscale x 8 x i1> @vscale_version() {
  %splatter = insertelement <vscale x 8 x i1> undef, i1 true, i32 0
  %foo = shufflevector <vscale x 8 x i1> %splatter,
                       <vscale x 8 x i1> undef,
                       <vscale x 8 x i32> zeroinitializer
  ret <vscale x 8 x i1> %foo
}

; The non-scalable version should be optimized as normal.

; CHECK-LABEL: define <8 x i1> @fixed_length_version() {
; CHECK-NEXT:  ret <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>
define <8 x i1> @fixed_length_version() {
  %splatter = insertelement <8 x i1> undef, i1 true, i32 0
  %foo = shufflevector <8 x i1> %splatter,
                       <8 x i1> undef,
                       <8 x i32> zeroinitializer
  ret <8 x i1> %foo
}

