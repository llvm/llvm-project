; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3 %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

@Ints9    = internal addrspace(10) global [9 x i32] poison
@Bools9   = internal addrspace(10) global [9 x i32] poison
@Ints12   = internal addrspace(10) global [12 x i32] poison
@Ints12B  = internal addrspace(10) global [12 x i32] poison
@Bools12  = internal addrspace(10) global [12 x i32] poison
@Floats12 = internal addrspace(10) global [12 x float] poison
@Floats12B = internal addrspace(10) global [12 x float] poison
@Ints16   = internal addrspace(10) global [16 x i32] poison
@Bools16  = internal addrspace(10) global [16 x i32] poison

; CHECK-DAG: %[[Bool:[0-9]+]] = OpTypeBool
; CHECK-DAG: %[[Int32:[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: %[[Vec4Int32:[0-9]+]] = OpTypeVector %[[Int32]] 4
; CHECK-DAG: %[[Vec4Bool:[0-9]+]] = OpTypeVector %[[Bool]] 4
; CHECK-DAG: %[[Float32:[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: %[[Vec4Float32:[0-9]+]] = OpTypeVector %[[Float32]] 4

; No vector wider than 4 lanes is ever materialized for shader targets.
; CHECK-NOT: OpTypeVector %[[Int32]] 8
; CHECK-NOT: OpTypeVector %[[Int32]] 9
; CHECK-NOT: OpTypeVector %[[Int32]] 12
; CHECK-NOT: OpTypeVector %[[Int32]] 16

;--- G_LOAD/G_STORE: always scalarized, even for the pow2-sized 4x4 case ---

; CHECK-LABEL: ; -- Begin function copy_bool3x3
; CHECK-COUNT-9: OpLoad %[[Int32]]
; CHECK-COUNT-9: OpStore
define internal void @copy_bool3x3() {
  %m = load <9 x i32>, ptr addrspace(10) @Ints9
  store <9 x i32> %m, ptr addrspace(10) @Bools9
  ret void
}

; Stands in for both bool3x4 and bool4x3 (both flatten to 12 elements).
; CHECK-LABEL: ; -- Begin function copy_bool_12elem
; CHECK-COUNT-12: OpLoad %[[Int32]]
; CHECK-COUNT-12: OpStore
define internal void @copy_bool_12elem() {
  %m = load <12 x i32>, ptr addrspace(10) @Ints12
  store <12 x i32> %m, ptr addrspace(10) @Bools12
  ret void
}

; CHECK-LABEL: ; -- Begin function copy_bool4x4
; CHECK-COUNT-16: OpLoad %[[Int32]]
; CHECK-COUNT-16: OpStore
define internal void @copy_bool4x4() {
  %m = load <16 x i32>, ptr addrspace(10) @Ints16
  store <16 x i32> %m, ptr addrspace(10) @Bools16
  ret void
}

;--- G_TRUNC + G_ZEXT: trunc-to-i1 then zext folds to (x & 1) ---
;
; The wide vector is rebuilt as N <= 4-lane chunks (all N OpCompositeConstruct
; first), then each chunk is masked with its own OpBitwiseAnd (all N masks
; after). Each mask's chunk operand is named and checked explicitly, in the
; same order codegen actually produces them, so the dataflow is tracked
; rather than counted blindly.

; CHECK-LABEL: ; -- Begin function bool3x3_zext
; CHECK: %[[C0:[0-9]+]] = OpCompositeConstruct %[[Vec4Int32]]
; CHECK: %[[C1:[0-9]+]] = OpCompositeConstruct %[[Vec4Int32]]
; CHECK: %[[C2:[0-9]+]] = OpCompositeConstruct %[[Vec4Int32]]
; CHECK: %{{[0-9]+}} = OpBitwiseAnd %[[Vec4Int32]] %[[C0]]
; CHECK: %{{[0-9]+}} = OpBitwiseAnd %[[Vec4Int32]] %[[C1]]
; CHECK: %{{[0-9]+}} = OpBitwiseAnd %[[Vec4Int32]] %[[C2]]
; CHECK-NOT: OpBitwiseAnd %[[Vec4Int32]]
define internal void @bool3x3_zext() {
  %m = load <9 x i32>, ptr addrspace(10) @Ints9
  %bits = trunc <9 x i32> %m to <9 x i1>
  %ext = zext <9 x i1> %bits to <9 x i32>
  store <9 x i32> %ext, ptr addrspace(10) @Bools9
  ret void
}

; Stands in for both bool3x4 and bool4x3.
; CHECK-LABEL: ; -- Begin function bool_12elem_zext
; CHECK: %[[C0:[0-9]+]] = OpCompositeConstruct %[[Vec4Int32]]
; CHECK: %[[C1:[0-9]+]] = OpCompositeConstruct %[[Vec4Int32]]
; CHECK: %[[C2:[0-9]+]] = OpCompositeConstruct %[[Vec4Int32]]
; CHECK: %{{[0-9]+}} = OpBitwiseAnd %[[Vec4Int32]] %[[C0]]
; CHECK: %{{[0-9]+}} = OpBitwiseAnd %[[Vec4Int32]] %[[C1]]
; CHECK: %{{[0-9]+}} = OpBitwiseAnd %[[Vec4Int32]] %[[C2]]
; CHECK-NOT: OpBitwiseAnd %[[Vec4Int32]]
define internal void @bool_12elem_zext() {
  %m = load <12 x i32>, ptr addrspace(10) @Ints12
  %bits = trunc <12 x i32> %m to <12 x i1>
  %ext = zext <12 x i1> %bits to <12 x i32>
  store <12 x i32> %ext, ptr addrspace(10) @Bools12
  ret void
}

; CHECK-LABEL: ; -- Begin function bool4x4_zext
; CHECK: %[[C0:[0-9]+]] = OpCompositeConstruct %[[Vec4Int32]]
; CHECK: %[[C1:[0-9]+]] = OpCompositeConstruct %[[Vec4Int32]]
; CHECK: %[[C2:[0-9]+]] = OpCompositeConstruct %[[Vec4Int32]]
; CHECK: %[[C3:[0-9]+]] = OpCompositeConstruct %[[Vec4Int32]]
; CHECK: %{{[0-9]+}} = OpBitwiseAnd %[[Vec4Int32]] %[[C0]]
; CHECK: %{{[0-9]+}} = OpBitwiseAnd %[[Vec4Int32]] %[[C1]]
; CHECK: %{{[0-9]+}} = OpBitwiseAnd %[[Vec4Int32]] %[[C2]]
; CHECK: %{{[0-9]+}} = OpBitwiseAnd %[[Vec4Int32]] %[[C3]]
; CHECK-NOT: OpBitwiseAnd %[[Vec4Int32]]
define internal void @bool4x4_zext() {
  %m = load <16 x i32>, ptr addrspace(10) @Ints16
  %bits = trunc <16 x i32> %m to <16 x i1>
  %ext = zext <16 x i1> %bits to <16 x i32>
  store <16 x i32> %ext, ptr addrspace(10) @Bools16
  ret void
}

;--- G_TRUNC + G_SEXT: trunc-to-i1 then sext canonicalizes to G_SEXT_INREG ---
;
; This target lowers G_SEXT_INREG to a shl/ashr pair by (bitwidth - 1), one
; pair per chunk, with each pair emitted back-to-back (unlike the zext masks
; above). Every chunk's shift-left-feeding-shift-right pair is named and
; checked explicitly below.

; CHECK-LABEL: ; -- Begin function bool3x3_sext
; CHECK: %[[Shl0:[0-9]+]] = OpShiftLeftLogical %[[Vec4Int32]]
; CHECK-NEXT: %{{[0-9]+}} = OpShiftRightArithmetic %[[Vec4Int32]] %[[Shl0]]
; CHECK: %[[Shl1:[0-9]+]] = OpShiftLeftLogical %[[Vec4Int32]]
; CHECK-NEXT: %{{[0-9]+}} = OpShiftRightArithmetic %[[Vec4Int32]] %[[Shl1]]
; CHECK: %[[Shl2:[0-9]+]] = OpShiftLeftLogical %[[Vec4Int32]]
; CHECK-NEXT: %{{[0-9]+}} = OpShiftRightArithmetic %[[Vec4Int32]] %[[Shl2]]
; CHECK-NOT: OpShiftLeftLogical %[[Vec4Int32]]
define internal void @bool3x3_sext() {
  %m = load <9 x i32>, ptr addrspace(10) @Ints9
  %bits = trunc <9 x i32> %m to <9 x i1>
  %ext = sext <9 x i1> %bits to <9 x i32>
  store <9 x i32> %ext, ptr addrspace(10) @Bools9
  ret void
}

; Stands in for both bool3x4 and bool4x3.
; CHECK-LABEL: ; -- Begin function bool_12elem_sext
; CHECK: %[[Shl0:[0-9]+]] = OpShiftLeftLogical %[[Vec4Int32]]
; CHECK-NEXT: %{{[0-9]+}} = OpShiftRightArithmetic %[[Vec4Int32]] %[[Shl0]]
; CHECK: %[[Shl1:[0-9]+]] = OpShiftLeftLogical %[[Vec4Int32]]
; CHECK-NEXT: %{{[0-9]+}} = OpShiftRightArithmetic %[[Vec4Int32]] %[[Shl1]]
; CHECK: %[[Shl2:[0-9]+]] = OpShiftLeftLogical %[[Vec4Int32]]
; CHECK-NEXT: %{{[0-9]+}} = OpShiftRightArithmetic %[[Vec4Int32]] %[[Shl2]]
; CHECK-NOT: OpShiftLeftLogical %[[Vec4Int32]]
define internal void @bool_12elem_sext() {
  %m = load <12 x i32>, ptr addrspace(10) @Ints12
  %bits = trunc <12 x i32> %m to <12 x i1>
  %ext = sext <12 x i1> %bits to <12 x i32>
  store <12 x i32> %ext, ptr addrspace(10) @Bools12
  ret void
}

; CHECK-LABEL: ; -- Begin function bool4x4_sext
; CHECK: %[[Shl0:[0-9]+]] = OpShiftLeftLogical %[[Vec4Int32]]
; CHECK-NEXT: %{{[0-9]+}} = OpShiftRightArithmetic %[[Vec4Int32]] %[[Shl0]]
; CHECK: %[[Shl1:[0-9]+]] = OpShiftLeftLogical %[[Vec4Int32]]
; CHECK-NEXT: %{{[0-9]+}} = OpShiftRightArithmetic %[[Vec4Int32]] %[[Shl1]]
; CHECK: %[[Shl2:[0-9]+]] = OpShiftLeftLogical %[[Vec4Int32]]
; CHECK-NEXT: %{{[0-9]+}} = OpShiftRightArithmetic %[[Vec4Int32]] %[[Shl2]]
; CHECK: %[[Shl3:[0-9]+]] = OpShiftLeftLogical %[[Vec4Int32]]
; CHECK-NEXT: %{{[0-9]+}} = OpShiftRightArithmetic %[[Vec4Int32]] %[[Shl3]]
; CHECK-NOT: OpShiftLeftLogical %[[Vec4Int32]]
define internal void @bool4x4_sext() {
  %m = load <16 x i32>, ptr addrspace(10) @Ints16
  %bits = trunc <16 x i32> %m to <16 x i1>
  %ext = sext <16 x i1> %bits to <16 x i32>
  store <16 x i32> %ext, ptr addrspace(10) @Bools16
  ret void
}

;--- Non-bool G_TRUNC/G_ZEXT/G_SEXT: same chunking, wider element type ---
;
; trunc-to-i16 then zext/sext-to-i32 folds the same way the i1 case does
; (mask-with-0xFFFF for zext, shl/ashr-by-16 for sext), so this checks the
; chunk splitting is independent of the truncated width, not just i1.

; CHECK-LABEL: ; -- Begin function narrow_12elem_zext
; CHECK: %[[C0:[0-9]+]] = OpCompositeConstruct %[[Vec4Int32]]
; CHECK: %[[C1:[0-9]+]] = OpCompositeConstruct %[[Vec4Int32]]
; CHECK: %[[C2:[0-9]+]] = OpCompositeConstruct %[[Vec4Int32]]
; CHECK: %{{[0-9]+}} = OpBitwiseAnd %[[Vec4Int32]] %[[C0]]
; CHECK: %{{[0-9]+}} = OpBitwiseAnd %[[Vec4Int32]] %[[C1]]
; CHECK: %{{[0-9]+}} = OpBitwiseAnd %[[Vec4Int32]] %[[C2]]
; CHECK-NOT: OpBitwiseAnd %[[Vec4Int32]]
define internal void @narrow_12elem_zext() {
  %m = load <12 x i32>, ptr addrspace(10) @Ints12
  %narrow = trunc <12 x i32> %m to <12 x i16>
  %wide = zext <12 x i16> %narrow to <12 x i32>
  store <12 x i32> %wide, ptr addrspace(10) @Bools12
  ret void
}

; CHECK-LABEL: ; -- Begin function narrow_16elem_sext
; CHECK: %[[Shl0:[0-9]+]] = OpShiftLeftLogical %[[Vec4Int32]]
; CHECK-NEXT: %{{[0-9]+}} = OpShiftRightArithmetic %[[Vec4Int32]] %[[Shl0]]
; CHECK: %[[Shl1:[0-9]+]] = OpShiftLeftLogical %[[Vec4Int32]]
; CHECK-NEXT: %{{[0-9]+}} = OpShiftRightArithmetic %[[Vec4Int32]] %[[Shl1]]
; CHECK: %[[Shl2:[0-9]+]] = OpShiftLeftLogical %[[Vec4Int32]]
; CHECK-NEXT: %{{[0-9]+}} = OpShiftRightArithmetic %[[Vec4Int32]] %[[Shl2]]
; CHECK: %[[Shl3:[0-9]+]] = OpShiftLeftLogical %[[Vec4Int32]]
; CHECK-NEXT: %{{[0-9]+}} = OpShiftRightArithmetic %[[Vec4Int32]] %[[Shl3]]
; CHECK-NOT: OpShiftLeftLogical %[[Vec4Int32]]
define internal void @narrow_16elem_sext() {
  %m = load <16 x i32>, ptr addrspace(10) @Ints16
  %narrow = trunc <16 x i32> %m to <16 x i16>
  %wide = sext <16 x i16> %narrow to <16 x i32>
  store <16 x i32> %wide, ptr addrspace(10) @Bools16
  ret void
}

;--- G_ICMP/G_FCMP: split the flattened matrix into 4-lane comparisons ---

; CHECK-LABEL: ; -- Begin function icmp_12elem
; CHECK-COUNT-3: OpIEqual %[[Vec4Bool]] %{{[0-9]+}} %{{[0-9]+}}
; CHECK-NOT: OpIEqual
define internal void @icmp_12elem() {
  %a = load <12 x i32>, ptr addrspace(10) @Ints12
  %b = load <12 x i32>, ptr addrspace(10) @Ints12B
  %cmp = icmp eq <12 x i32> %a, %b
  %ext = zext <12 x i1> %cmp to <12 x i32>
  store <12 x i32> %ext, ptr addrspace(10) @Bools12
  ret void
}

; CHECK-LABEL: ; -- Begin function fcmp_12elem
; CHECK-COUNT-3: OpFOrdEqual %[[Vec4Bool]] %{{[0-9]+}} %{{[0-9]+}}
; CHECK-NOT: OpFOrdEqual
define internal void @fcmp_12elem() {
  %a = load <12 x float>, ptr addrspace(10) @Floats12
  %b = load <12 x float>, ptr addrspace(10) @Floats12B
  %cmp = fcmp oeq <12 x float> %a, %b
  %ext = zext <12 x i1> %cmp to <12 x i32>
  store <12 x i32> %ext, ptr addrspace(10) @Bools12
  ret void
}

define void @main() #0 {
  call void @copy_bool3x3()
  call void @copy_bool_12elem()
  call void @copy_bool4x4()
  call void @bool3x3_zext()
  call void @bool_12elem_zext()
  call void @bool4x4_zext()
  call void @bool3x3_sext()
  call void @bool_12elem_sext()
  call void @bool4x4_sext()
  call void @narrow_12elem_zext()
  call void @narrow_16elem_sext()
  call void @icmp_12elem()
  call void @fcmp_12elem()
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
