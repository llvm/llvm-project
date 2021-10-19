; RUN: llc -aarch64-sve-vector-bits-min=256 < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

; Currently there is no custom lowering for vector shuffles operating on types
; bigger than NEON. However, having no support opens us up to a code generator
; hang when expanding BUILD_VECTOR. Here we just validate the promblematic case
; successfully exits code generation.
define void @hang_when_merging_stores_after_legalisation(<8 x i32>* %a, <2 x i32> %b) #0 {
; CHECK-LABEL: hang_when_merging_stores_after_legalisation:
  %splat = shufflevector <2 x i32> %b, <2 x i32> undef, <8 x i32> zeroinitializer
  %interleaved.vec = shufflevector <8 x i32> %splat, <8 x i32> undef, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x i32> %interleaved.vec, <8 x i32>* %a, align 4
  ret void
}

attributes #0 = { nounwind "target-features"="+sve" }
