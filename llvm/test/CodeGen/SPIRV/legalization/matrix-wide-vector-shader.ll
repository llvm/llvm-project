; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3 %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; Wide/non-standard-width vector legalization for LLVM matrix types on
; shader/Vulkan targets, where SPIR-V vectors are limited to 4 components.
; A bool3x4 flattens to a 12-element vector and a bool3x3 to a 9-element
; vector; loads, stores, truncations, zero/sign extends of those must be split
; into <= 4 lane chunks and never materialize a vector wider than 4.
;

@in12  = internal addrspace(10) global [12 x i32] poison
@out12 = internal addrspace(10) global [12 x i32] poison
@in9   = internal addrspace(10) global [9 x i32] poison
@out9  = internal addrspace(10) global [9 x i32] poison

; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#V4Int:]] = OpTypeVector %[[#Int]] 4
; CHECK-DAG: %[[#V3Int:]] = OpTypeVector %[[#Int]] 3

; No vector wider than 4 components may ever be produced for shader targets.
; CHECK-NOT: OpTypeVector %[[#Int]] 12
; CHECK-NOT: OpTypeVector %[[#Int]] 9
; CHECK-NOT: OpTypeVector %[[#Int]] 8

;===----------------------------------------------------------------------===;
; Pure load + store (no extend) of the flattened matrix.
;===----------------------------------------------------------------------===;

; CHECK-LABEL: ; -- Begin function load_store_3x4
; The wide load/store is scalarized into twelve per-element scalar accesses.
; CHECK-COUNT-12: OpLoad %[[#Int]]
; CHECK-COUNT-12: OpStore
define internal void @load_store_3x4() {
  %v = load <12 x i32>, ptr addrspace(10) @in12
  store <12 x i32> %v, ptr addrspace(10) @out12
  ret void
}

; CHECK-LABEL: ; -- Begin function load_store_3x3
; CHECK-COUNT-9: OpLoad %[[#Int]]
; CHECK-COUNT-9: OpStore
define internal void @load_store_3x3() {
  %v = load <9 x i32>, ptr addrspace(10) @in9
  store <9 x i32> %v, ptr addrspace(10) @out9
  ret void
}

; A 4x3 matrix flattens to the same 12-element width as 3x4, so it legalizes
; identically: twelve per-element scalar load/stores.
; CHECK-LABEL: ; -- Begin function load_store_4x3
; CHECK-COUNT-12: OpLoad %[[#Int]]
; CHECK-COUNT-12: OpStore
define internal void @load_store_4x3() {
  %v = load <12 x i32>, ptr addrspace(10) @in12
  store <12 x i32> %v, ptr addrspace(10) @out12
  ret void
}

;===----------------------------------------------------------------------===;
; Truncation to i1 (bool matrix element) then zero-extend back to i32.
; trunc(x)->i1 followed by zext(->i32) folds to (x & 1), so no INotEqual is
; produced. Both flattened widths split into uniform vector chunks: the 12-lane
; case as three <4 x i32> chunks, the 9-lane case as three <3 x i32> chunks.
;===----------------------------------------------------------------------===;

; CHECK-LABEL: ; -- Begin function trunc_zext_3x4
; CHECK-COUNT-3: OpBitwiseAnd %[[#V4Int]]
; CHECK-NOT: OpBitwiseAnd %[[#V4Int]]
define internal void @trunc_zext_3x4() {
  %v = load <12 x i32>, ptr addrspace(10) @in12
  %b = trunc <12 x i32> %v to <12 x i1>
  %e = zext <12 x i1> %b to <12 x i32>
  store <12 x i32> %e, ptr addrspace(10) @out12
  ret void
}

; The 9-lane width shares divisor 3 with the max vector size, so it splits into
; three <3 x i32> chunks (not scalarized).
; CHECK-LABEL: ; -- Begin function trunc_zext_3x3
; CHECK-COUNT-3: OpBitwiseAnd %[[#V3Int]]
; CHECK-NOT: OpBitwiseAnd %[[#V3Int]]
define internal void @trunc_zext_3x3() {
  %v = load <9 x i32>, ptr addrspace(10) @in9
  %b = trunc <9 x i32> %v to <9 x i1>
  %e = zext <9 x i1> %b to <9 x i32>
  store <9 x i32> %e, ptr addrspace(10) @out9
  ret void
}

; A 4x3 matrix is 12 elements, so it splits into three <4 x i32> chunks like 3x4.
; CHECK-LABEL: ; -- Begin function trunc_zext_4x3
; CHECK-COUNT-3: OpBitwiseAnd %[[#V4Int]]
; CHECK-NOT: OpBitwiseAnd %[[#V4Int]]
define internal void @trunc_zext_4x3() {
  %v = load <12 x i32>, ptr addrspace(10) @in12
  %b = trunc <12 x i32> %v to <12 x i1>
  %e = zext <12 x i1> %b to <12 x i32>
  store <12 x i32> %e, ptr addrspace(10) @out12
  ret void
}

;===----------------------------------------------------------------------===;
; Truncation to i1 then sign-extend back to i32.
; sext(trunc(x)->i1) canonicalizes to G_SEXT_INREG, which this target lowers
; to the canonical shl/ashr pair (shift left then arithmetic shift right by
; the bit-width minus one), so no boolean type is ever materialized. Both
; widths split the same GCD-divisor way as trunc_zext above: the 12-lane case
; as three <4 x i32> chunks, the 9-lane case as three <3 x i32> chunks.
;===----------------------------------------------------------------------===;

; CHECK-LABEL: ; -- Begin function trunc_sext_3x4
; CHECK: OpShiftLeftLogical %[[#V4Int]]
; CHECK-NEXT: OpShiftRightArithmetic %[[#V4Int]]
; CHECK: OpShiftLeftLogical %[[#V4Int]]
; CHECK-NEXT: OpShiftRightArithmetic %[[#V4Int]]
; CHECK: OpShiftLeftLogical %[[#V4Int]]
; CHECK-NEXT: OpShiftRightArithmetic %[[#V4Int]]
; CHECK-NOT: OpShiftLeftLogical %[[#V4Int]]
define internal void @trunc_sext_3x4() {
  %v = load <12 x i32>, ptr addrspace(10) @in12
  %b = trunc <12 x i32> %v to <12 x i1>
  %e = sext <12 x i1> %b to <12 x i32>
  store <12 x i32> %e, ptr addrspace(10) @out12
  ret void
}

; The 9-lane width shares divisor 3 with the max vector size, so it splits
; into three <3 x i32> chunks (not scalarized), same as trunc_zext_3x3.
; CHECK-LABEL: ; -- Begin function trunc_sext_3x3
; CHECK: OpShiftLeftLogical %[[#V3Int]]
; CHECK-NEXT: OpShiftRightArithmetic %[[#V3Int]]
; CHECK: OpShiftLeftLogical %[[#V3Int]]
; CHECK-NEXT: OpShiftRightArithmetic %[[#V3Int]]
; CHECK: OpShiftLeftLogical %[[#V3Int]]
; CHECK-NEXT: OpShiftRightArithmetic %[[#V3Int]]
; CHECK-NOT: OpShiftLeftLogical %[[#V3Int]]
define internal void @trunc_sext_3x3() {
  %v = load <9 x i32>, ptr addrspace(10) @in9
  %b = trunc <9 x i32> %v to <9 x i1>
  %e = sext <9 x i1> %b to <9 x i32>
  store <9 x i32> %e, ptr addrspace(10) @out9
  ret void
}

; A 4x3 matrix is 12 elements, so it stays vectorized as three <4 x> chunks.
; CHECK-LABEL: ; -- Begin function trunc_sext_4x3
; CHECK: OpShiftLeftLogical %[[#V4Int]]
; CHECK-NEXT: OpShiftRightArithmetic %[[#V4Int]]
; CHECK: OpShiftLeftLogical %[[#V4Int]]
; CHECK-NEXT: OpShiftRightArithmetic %[[#V4Int]]
; CHECK: OpShiftLeftLogical %[[#V4Int]]
; CHECK-NEXT: OpShiftRightArithmetic %[[#V4Int]]
; CHECK-NOT: OpShiftLeftLogical %[[#V4Int]]
define internal void @trunc_sext_4x3() {
  %v = load <12 x i32>, ptr addrspace(10) @in12
  %b = trunc <12 x i32> %v to <12 x i1>
  %e = sext <12 x i1> %b to <12 x i32>
  store <12 x i32> %e, ptr addrspace(10) @out12
  ret void
}

define void @main() #0 {
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
