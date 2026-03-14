; RUN: llvm-dis < %s.bc | FileCheck %s

; Bitcode was generated from file below

define <2 x float> @test_vbfdot_f32(<2 x float> %r, <4 x bfloat> %a, <4 x bfloat> %b) {
; CHECK-LABEL: @test_vbfdot_f32
entry:
  %0 = bitcast <4 x bfloat> %a to <8 x i8>
  %1 = bitcast <4 x bfloat> %b to <8 x i8>
  ; CHECK: %2 = bitcast <8 x i8> %0 to <4 x bfloat>
  ; CHECK-NEXT: %3 = bitcast <8 x i8> %1 to <4 x bfloat>
  ; CHECK-NEXT: %vbfdot1.i = call <2 x float> @llvm.aarch64.neon.bfdot.v2f32.v4bf16(<2 x float> %r, <4 x bfloat> %2, <4 x bfloat> %3)
  %vbfdot1.i = call <2 x float> @llvm.aarch64.neon.bfdot.v2f32.v8i8(<2 x float> %r, <8 x i8> %0, <8 x i8> %1)
  ret <2 x float> %vbfdot1.i
}

define <4 x float> @test_vbfdotq_f32(<4 x float> %r, <8 x bfloat> %a, <8 x bfloat> %b) {
; CHECK-LABEL: @test_vbfdotq_f32
entry:
  %0 = bitcast <8 x bfloat> %a to <16 x i8>
  %1 = bitcast <8 x bfloat> %b to <16 x i8>
  ; CHECK: %2 = bitcast <16 x i8> %0 to <8 x bfloat>
  ; CHECK-NEXT: %3 = bitcast <16 x i8> %1 to <8 x bfloat>
  ; CHECK-NEXT: %vbfdot1.i = call <4 x float> @llvm.aarch64.neon.bfdot.v4f32.v8bf16(<4 x float> %r, <8 x bfloat> %2, <8 x bfloat> %3)
  %vbfdot1.i = call <4 x float> @llvm.aarch64.neon.bfdot.v4f32.v16i8(<4 x float> %r, <16 x i8> %0, <16 x i8> %1)
  ret <4 x float> %vbfdot1.i
}

define <4 x float> @test_vbfmmlaq_f32(<4 x float> %r, <8 x bfloat> %a, <8 x bfloat> %b) {
; CHECK-LABEL: @test_vbfmmlaq_f32
entry:
  %0 = bitcast <8 x bfloat> %a to <16 x i8>
  %1 = bitcast <8 x bfloat> %b to <16 x i8>
  %vbfmmla1.i = call <4 x float> @llvm.aarch64.neon.bfmmla.v4f32.v16i8(<4 x float> %r, <16 x i8> %0, <16 x i8> %1)
  ; CHECK: %2 = bitcast <16 x i8> %0 to <8 x bfloat>
  ; CHECK-NEXT: %3 = bitcast <16 x i8> %1 to <8 x bfloat>
  ; CHECK-NEXT: %vbfmmla1.i = call <4 x float> @llvm.aarch64.neon.bfmmla(<4 x float> %r, <8 x bfloat> %2, <8 x bfloat> %3)
  ret <4 x float> %vbfmmla1.i
}

define <4 x float> @test_vbfmlalbq_laneq_f32(<4 x float> %r, <8 x bfloat> %a, <8 x bfloat> %b) {
; CHECK-LABEL: @test_vbfmlalbq_laneq_f32
entry:
  %vecinit35 = shufflevector <8 x bfloat> %b, <8 x bfloat> undef, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
  %0 = bitcast <8 x bfloat> %a to <16 x i8>
  %1 = bitcast <8 x bfloat> %vecinit35 to <16 x i8>
  %vbfmlalb1.i = call <4 x float> @llvm.aarch64.neon.bfmlalb.v4f32.v16i8(<4 x float> %r, <16 x i8> %0, <16 x i8> %1)
  ; CHECK: %2 = bitcast <16 x i8> %0 to <8 x bfloat>
  ; CHECK-NEXT: %3 = bitcast <16 x i8> %1 to <8 x bfloat>
  ; CHECK-NEXT: %vbfmlalb1.i = call <4 x float> @llvm.aarch64.neon.bfmlalb(<4 x float> %r, <8 x bfloat> %2, <8 x bfloat> %3)
  ret <4 x float> %vbfmlalb1.i
}

define <4 x float> @test_vbfmlaltq_laneq_f32(<4 x float> %r, <8 x bfloat> %a, <8 x bfloat> %b) {
; CHECK-LABEL: @test_vbfmlaltq_laneq_f32
entry:
  %vecinit35 = shufflevector <8 x bfloat> %b, <8 x bfloat> undef, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
  %0 = bitcast <8 x bfloat> %a to <16 x i8>
  %1 = bitcast <8 x bfloat> %vecinit35 to <16 x i8>
  %vbfmlalt1.i = call <4 x float> @llvm.aarch64.neon.bfmlalt.v4f32.v16i8(<4 x float> %r, <16 x i8> %0, <16 x i8> %1)
  ; CHECK: %2 = bitcast <16 x i8> %0 to <8 x bfloat>
  ; CHECK-NEXT: %3 = bitcast <16 x i8> %1 to <8 x bfloat>
  ; CHECK-NEXT: %vbfmlalt1.i = call <4 x float> @llvm.aarch64.neon.bfmlalt(<4 x float> %r, <8 x bfloat> %2, <8 x bfloat> %3)
  ret <4 x float> %vbfmlalt1.i
}

declare <2 x float> @llvm.aarch64.neon.bfdot.v2f32.v8i8(<2 x float>, <8 x i8>, <8 x i8>)
; CHECK: declare <2 x float> @llvm.aarch64.neon.bfdot.v2f32.v4bf16(<2 x float>, <4 x bfloat>, <4 x bfloat>)
declare <4 x float> @llvm.aarch64.neon.bfdot.v4f32.v16i8(<4 x float>, <16 x i8>, <16 x i8>)
; CHECK: declare <4 x float> @llvm.aarch64.neon.bfdot.v4f32.v8bf16(<4 x float>, <8 x bfloat>, <8 x bfloat>)
declare <4 x float> @llvm.aarch64.neon.bfmmla.v4f32.v16i8(<4 x float>, <16 x i8>, <16 x i8>)
; CHECK: declare <4 x float> @llvm.aarch64.neon.bfmmla(<4 x float>, <8 x bfloat>, <8 x bfloat>)
declare <4 x float> @llvm.aarch64.neon.bfmlalb.v4f32.v16i8(<4 x float>, <16 x i8>, <16 x i8>)
; CHECK: declare <4 x float> @llvm.aarch64.neon.bfmlalb(<4 x float>, <8 x bfloat>, <8 x bfloat>)
declare <4 x float> @llvm.aarch64.neon.bfmlalt.v4f32.v16i8(<4 x float>, <16 x i8>, <16 x i8>)
; CHECK: declare <4 x float> @llvm.aarch64.neon.bfmlalt(<4 x float>, <8 x bfloat>, <8 x bfloat>)