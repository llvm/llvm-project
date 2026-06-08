; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NOEXT
; RUN: %if spirv-tools %{ llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown %s --spirv-ext=+SPV_ALTERA_arbitrary_precision_integers -o - | FileCheck %s --check-prefixes=CHECK,CHECK-EXT
; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s --spirv-ext=+SPV_ALTERA_arbitrary_precision_integers -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NOEXT
; RUN: %if spirv-tools %{ llc -O0 -verify-machineinstrs -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv32-unknown-unknown %s --spirv-ext=+SPV_ALTERA_arbitrary_precision_integers -o - | FileCheck %s --check-prefixes=CHECK,CHECK-EXT
; RUN: %if spirv-tools %{ llc -O0 -verify-machineinstrs -mtriple=spirv32-unknown-unknown %s --spirv-ext=+SPV_ALTERA_arbitrary_precision_integers -o - -filetype=obj | spirv-val %}

; CHECK-EXT-DAG: %[[#Int40:]] = OpTypeInt 40 0
; CHECK-EXT-DAG: %[[#Int50:]] = OpTypeInt 50 0
; CHECK-EXT-DAG: %[[#Int24:]] = OpTypeInt 24 0
; CHECK-EXT-DAG: %[[#ExtInt32:]] = OpTypeInt 32 0
; CHECK-NOEXT-DAG: %[[#Int64:]] = OpTypeInt 64 0
; CHECK-NOEXT-DAG: %[[#Int32:]] = OpTypeInt 32 0
; CHECK-EXT-DAG: %[[#Vec2Int40:]] = OpTypeVector %[[#Int40]] 2
; CHECK-EXT-DAG: %[[#Vec2Int50:]] = OpTypeVector %[[#Int50]] 2
; CHECK-EXT-DAG: %[[#Vec2ExtInt32:]] = OpTypeVector %[[#ExtInt32]] 2
; CHECK-EXT-DAG: %[[#Vec3Int50:]] = OpTypeVector %[[#Int50]] 3
; CHECK-EXT-DAG: %[[#Vec3Int24:]] = OpTypeVector %[[#Int24]] 3
; CHECK-EXT-DAG: %[[#Vec4Int24:]] = OpTypeVector %[[#Int24]] 4
; CHECK-NOEXT-DAG: %[[#Vec2Int64:]] = OpTypeVector %[[#Int64]] 2
; CHECK-NOEXT-DAG: %[[#Vec2Int32:]] = OpTypeVector %[[#Int32]] 2
; CHECK-NOEXT-DAG: %[[#Vec3Int64:]] = OpTypeVector %[[#Int64]] 3
; CHECK-NOEXT-DAG: %[[#Vec4Int64:]] = OpTypeVector %[[#Int64]] 4
; CHECK-NOEXT-DAG: %[[#Vec3Int32:]] = OpTypeVector %[[#Int32]] 3
; CHECK-NOEXT-DAG: %[[#Vec4Int32:]] = OpTypeVector %[[#Int32]] 4
; CHECK-NOEXT-DAG: %[[#Mask40:]] = OpConstant %[[#Int64]] 1099511627775
; CHECK-NOEXT-DAG: %[[#Mask24:]] = OpConstant %[[#Int64]] 16777215
; CHECK-NOEXT-DAG: %[[#Mask40Vec2:]] = OpConstantComposite %[[#Vec2Int64]] %[[#Mask40]] %[[#Mask40]]
; CHECK-NOEXT-DAG: %[[#Mask24Vec3:]] = OpConstantComposite %[[#Vec3Int64]] %[[#Mask24]] %[[#Mask24]] %[[#Mask24]]
; CHECK-NOEXT-DAG: %[[#Mask24Vec4:]] = OpConstantComposite %[[#Vec4Int64]] %[[#Mask24]] %[[#Mask24]] %[[#Mask24]] %[[#Mask24]]


; Test i64 -> i40: both widen to i64
; CHECK: OpFunction
; CHECK: %[[#T1Arg:]] = OpFunctionParameter
; CHECK: %[[#T1Val:]] = OpFunctionParameter
; CHECK-EXT: %[[#T1Tr:]] = OpUConvert %[[#Int40]] %[[#T1Val]]
; CHECK-EXT: OpStore %[[#T1Arg]] %[[#T1Tr]]
; CHECK-NOEXT: %[[#T1And:]] = OpBitwiseAnd %[[#Int64]] %[[#T1Val]] %[[#]]
; CHECK-NOEXT: OpStore %[[#T1Arg]] %[[#T1And]]
define spir_kernel void @trunc_i64_to_i40(ptr addrspace(1) %arg, i64 %val) {
  %tr = trunc i64 %val to i40
  store i40 %tr, ptr addrspace(1) %arg
  ret void
}

; Test i50 -> i24: src widens to i64, dst widens to i32
; CHECK: OpFunction
; CHECK: %[[#T2Arg:]] = OpFunctionParameter
; CHECK: %[[#T2Val:]] = OpFunctionParameter
; CHECK-EXT: %[[#T2Tr:]] = OpUConvert %[[#Int24]] %[[#T2Val]]
; CHECK-EXT: OpStore %[[#T2Arg]] %[[#T2Tr]]
; CHECK-NOEXT: %[[#T2And:]] = OpBitwiseAnd %[[#Int64]] %[[#T2Val]] %[[#]]
; CHECK-NOEXT: %[[#T2Conv:]] = OpUConvert %[[#Int32]] %[[#T2And]]
; CHECK-NOEXT: OpStore %[[#T2Arg]] %[[#T2Conv]]
define spir_kernel void @trunc_i50_to_i24(ptr addrspace(1) %arg, i50 %val) {
  %tr = trunc i50 %val to i24
  store i24 %tr, ptr addrspace(1) %arg
  ret void
}

; Test i64 -> i24: src stays i64, dst widens to i32
; CHECK: OpFunction
; CHECK: %[[#T3Arg:]] = OpFunctionParameter
; CHECK: %[[#T3Val:]] = OpFunctionParameter
; CHECK-EXT: %[[#T3Tr:]] = OpUConvert %[[#Int24]] %[[#T3Val]]
; CHECK-EXT: OpStore %[[#T3Arg]] %[[#T3Tr]]
; CHECK-NOEXT: %[[#T3And:]] = OpBitwiseAnd %[[#Int64]] %[[#T3Val]] %[[#]]
; CHECK-NOEXT: %[[#T3Conv:]] = OpUConvert %[[#Int32]] %[[#T3And]]
; CHECK-NOEXT: OpStore %[[#T3Arg]] %[[#T3Conv]]
define spir_kernel void @trunc_i64_to_i24(ptr addrspace(1) %arg, i64 %val) {
  %tr = trunc i64 %val to i24
  store i24 %tr, ptr addrspace(1) %arg
  ret void
}

; Test <2 x i64> -> <2 x i40>: both widen to <2 x i64>
; CHECK: OpFunction
; CHECK: %[[#T4Arg:]] = OpFunctionParameter
; CHECK: %[[#T4Val:]] = OpFunctionParameter
; CHECK-EXT: %[[#T4Tr:]] = OpUConvert %[[#Vec2Int40]] %[[#T4Val]]
; CHECK-EXT: OpStore %[[#T4Arg]] %[[#T4Tr]]
; CHECK-NOEXT: %[[#T4And:]] = OpBitwiseAnd %[[#Vec2Int64]] %[[#T4Val]] %[[#Mask40Vec2]]
; CHECK-NOEXT: OpStore %[[#T4Arg]] %[[#T4And]]
define spir_kernel void @trunc_v2i64_to_v2i40(ptr addrspace(1) %arg, <2 x i64> %val) {
  %tr = trunc <2 x i64> %val to <2 x i40>
  store <2 x i40> %tr, ptr addrspace(1) %arg
  ret void
}

; Test <3 x i50> -> <3 x i24>: src widens to <3 x i64>, dst widens to <3 x i32>
; CHECK: OpFunction
; CHECK: %[[#T5Arg:]] = OpFunctionParameter
; CHECK: %[[#T5Val:]] = OpFunctionParameter
; CHECK-EXT: %[[#T5Tr:]] = OpUConvert %[[#Vec3Int24]] %[[#T5Val]]
; CHECK-EXT: OpStore %[[#T5Arg]] %[[#T5Tr]]
; CHECK-NOEXT: %[[#T5And:]] = OpBitwiseAnd %[[#Vec3Int64]] %[[#T5Val]] %[[#Mask24Vec3]]
; CHECK-NOEXT: %[[#T5Conv:]] = OpUConvert %[[#Vec3Int32]] %[[#T5And]]
; CHECK-NOEXT: OpStore %[[#T5Arg]] %[[#T5Conv]]
define spir_kernel void @trunc_v3i50_to_v3i24(ptr addrspace(1) %arg, <3 x i50> %val) {
  %tr = trunc <3 x i50> %val to <3 x i24>
  store <3 x i24> %tr, ptr addrspace(1) %arg
  ret void
}

; Test <4 x i64> -> <4 x i24>: src stays <4 x i64>, dst widens to <4 x i32>
; CHECK: OpFunction
; CHECK: %[[#T6Arg:]] = OpFunctionParameter
; CHECK: %[[#T6Val:]] = OpFunctionParameter
; CHECK-EXT: %[[#T6Tr:]] = OpUConvert %[[#Vec4Int24]] %[[#T6Val]]
; CHECK-EXT: OpStore %[[#T6Arg]] %[[#T6Tr]]
; CHECK-NOEXT: %[[#T6And:]] = OpBitwiseAnd %[[#Vec4Int64]] %[[#T6Val]] %[[#Mask24Vec4]]
; CHECK-NOEXT: %[[#T6Conv:]] = OpUConvert %[[#Vec4Int32]] %[[#T6And]]
; CHECK-NOEXT: OpStore %[[#T6Arg]] %[[#T6Conv]]
define spir_kernel void @trunc_v4i64_to_v4i24(ptr addrspace(1) %arg, <4 x i64> %val) {
  %tr = trunc <4 x i64> %val to <4 x i24>
  store <4 x i24> %tr, ptr addrspace(1) %arg
  ret void
}

; Test <2 x i50> -> <2 x i32>: dst width is already legal, no mask needed
; CHECK: OpFunction
; CHECK: %[[#T7Arg:]] = OpFunctionParameter
; CHECK: %[[#T7Val:]] = OpFunctionParameter
; CHECK-EXT: %[[#T7Tr:]] = OpUConvert %[[#Vec2ExtInt32]] %[[#T7Val]]
; CHECK-EXT: OpStore %[[#T7Arg]] %[[#T7Tr]]
; CHECK-NOEXT-NOT: OpBitwiseAnd
; CHECK-NOEXT: %[[#T7Conv:]] = OpUConvert %[[#Vec2Int32]] %[[#T7Val]]
; CHECK-NOEXT: OpStore %[[#T7Arg]] %[[#T7Conv]]
define spir_kernel void @trunc_v2i50_to_v2i32(ptr addrspace(1) %arg, <2 x i50> %val) {
  %tr = trunc <2 x i50> %val to <2 x i32>
  store <2 x i32> %tr, ptr addrspace(1) %arg
  ret void
}
