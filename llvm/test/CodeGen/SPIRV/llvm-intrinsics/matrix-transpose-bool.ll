; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3 %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; Regression test for https://github.com/llvm/llvm-project/issues/186864
;
; Transposing a boolean matrix (e.g. bool3x4 / bool3x3 in HLSL) flattens the
; matrix to a wide <N x i1> vector and pairs it with a trunc/zext to/from
; <N x i32>. On shader/Vulkan targets SPIR-V vectors are limited to 4
; components, so the wide G_TRUNC / G_ZEXT are split into <= 4 lane chunks by
; the legalizer. This test makes sure:
;   * no OpTypeVector wider than 4 components is ever materialized,
;   * the boolean trunc chunks are lowered with a *vector* bool result type
;     (OpINotEqual %<N x bool>), not a scalar bool, and
;   * the transpose shuffle stays vectorized (OpVectorShuffle on the legal
;     chunk width) so the post-transpose zext also stays vectorized
;     (OpSelect %<N x int>), rather than being scalarized to one select per
;     matrix element.

@in12 = internal addrspace(10) global [12 x i32] poison
@out12 = internal addrspace(10) global [12 x i32] poison
@in9 = internal addrspace(10) global [9 x i32] poison
@out9 = internal addrspace(10) global [9 x i32] poison

; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Bool:]] = OpTypeBool
; CHECK-DAG: %[[#V4Int:]] = OpTypeVector %[[#Int]] 4
; CHECK-DAG: %[[#V4Bool:]] = OpTypeVector %[[#Bool]] 4
; CHECK-DAG: %[[#V3Int:]] = OpTypeVector %[[#Int]] 3
; CHECK-DAG: %[[#V3Bool:]] = OpTypeVector %[[#Bool]] 3

; No vector wider than 4 components may be produced for shader targets.
; CHECK-NOT: OpTypeVector %[[#Int]] 12
; CHECK-NOT: OpTypeVector %[[#Bool]] 12
; CHECK-NOT: OpTypeVector %[[#Int]] 9
; CHECK-NOT: OpTypeVector %[[#Bool]] 9

; Test transpose of a bool 3x4 matrix (12 elements -> three <4 x i1> chunks).
; CHECK-LABEL: ; -- Begin function test_transpose_bool_3x4
; The wide trunc splits into exactly three <4 x i1> chunks: three vector
; INotEqual, each with a <4 x bool> result type (never scalar or wider).
; CHECK-COUNT-3: OpINotEqual %[[#V4Bool]]
; CHECK-NOT: OpINotEqual %[[#V4Bool]]
; The transpose is lowered as vector shuffles on the <4 x bool> chunks instead
; of being scalarized (nine OpVectorShuffle: three per <4 x> result chunk).
; CHECK-COUNT-9: OpVectorShuffle %[[#V4Bool]]
; CHECK-NOT: OpVectorShuffle %[[#V4Bool]]
; The post-transpose zext therefore stays vectorized: three <4 x int> selects.
; CHECK-COUNT-3: OpSelect %[[#V4Int]]
; CHECK-NOT: OpSelect
define internal void @test_transpose_bool_3x4() {
  %src = load <12 x i32>, ptr addrspace(10) @in12
  %b = trunc <12 x i32> %src to <12 x i1>
  %t = call <12 x i1> @llvm.matrix.transpose.v12i1(<12 x i1> %b, i32 3, i32 4)
  %ext = zext <12 x i1> %t to <12 x i32>
  store <12 x i32> %ext, ptr addrspace(10) @out12
  ret void
}

; Test transpose of a bool 3x3 matrix (9 elements -> three <3 x i1> chunks).
; The odd width shares no divisor > 1 with 4, but 3 is itself a legal SPIR-V
; vector size, so it splits into three <3 x> vectors rather than scalarizing.
; CHECK-LABEL: ; -- Begin function test_transpose_bool_3x3
; Three vector INotEqual, each with a <3 x bool> result type.
; CHECK-COUNT-3: OpINotEqual %[[#V3Bool]]
; CHECK-NOT: OpINotEqual %[[#V3Bool]]
; The transpose stays vectorized on the <3 x bool> chunks (nine OpVectorShuffle:
; three per <3 x> result chunk).
; CHECK-COUNT-9: OpVectorShuffle %[[#V3Bool]]
; CHECK-NOT: OpVectorShuffle %[[#V3Bool]]
; The post-transpose zext stays vectorized: three <3 x int> selects.
; CHECK-COUNT-3: OpSelect %[[#V3Int]]
; CHECK-NOT: OpSelect
define internal void @test_transpose_bool_3x3() {
  %src = load <9 x i32>, ptr addrspace(10) @in9
  %b = trunc <9 x i32> %src to <9 x i1>
  %t = call <9 x i1> @llvm.matrix.transpose.v9i1(<9 x i1> %b, i32 3, i32 3)
  %ext = zext <9 x i1> %t to <9 x i32>
  store <9 x i32> %ext, ptr addrspace(10) @out9
  ret void
}

; Test transpose of a bool 4x3 matrix (12 elements -> three <4 x i1> chunks).
; A 4x3 matrix has the same flattened width as 3x4, so it splits identically;
; only the transpose shuffle masks differ.
; CHECK-LABEL: ; -- Begin function test_transpose_bool_4x3
; The wide trunc splits into exactly three <4 x i1> chunks: three vector
; INotEqual, each with a <4 x bool> result type (never scalar or wider).
; CHECK-COUNT-3: OpINotEqual %[[#V4Bool]]
; CHECK-NOT: OpINotEqual %[[#V4Bool]]
; The transpose stays vectorized on the <4 x bool> chunks (nine OpVectorShuffle:
; three per <4 x> result chunk).
; CHECK-COUNT-9: OpVectorShuffle %[[#V4Bool]]
; CHECK-NOT: OpVectorShuffle %[[#V4Bool]]
; The post-transpose zext stays vectorized: three <4 x int> selects.
; CHECK-COUNT-3: OpSelect %[[#V4Int]]
; CHECK-NOT: OpSelect
define internal void @test_transpose_bool_4x3() {
  %src = load <12 x i32>, ptr addrspace(10) @in12
  %b = trunc <12 x i32> %src to <12 x i1>
  %t = call <12 x i1> @llvm.matrix.transpose.v12i1(<12 x i1> %b, i32 4, i32 3)
  %ext = zext <12 x i1> %t to <12 x i32>
  store <12 x i32> %ext, ptr addrspace(10) @out12
  ret void
}

define void @main() #0 {
  ret void
}

declare <12 x i1> @llvm.matrix.transpose.v12i1(<12 x i1>, i32, i32)
declare <9 x i1> @llvm.matrix.transpose.v9i1(<9 x i1>, i32, i32)

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
