// RUN: mlir-opt -mlir-disable-threading -test-spirv-op-availability %s | FileCheck %s

// CHECK-LABEL: iadd
func.func @iadd(%arg: i32) -> i32 {
  // CHECK: min version: v1.0
  // CHECK: max version: v1.6
  // CHECK: extensions: [ ]
  // CHECK: capabilities: [ ]
  %0 = spirv.IAdd %arg, %arg: i32
  return %0: i32
}

// CHECK: atomic_compare_exchange_weak
func.func @atomic_compare_exchange_weak(%ptr: !spirv.ptr<i32, Workgroup>, %value: i32, %comparator: i32) -> i32 {
  // CHECK: min version: v1.0
  // CHECK: max version: v1.3
  // CHECK: extensions: [ ]
  // CHECK: capabilities: [ [Kernel] ]
  %0 = spirv.AtomicCompareExchangeWeak <Workgroup> <Release> <Acquire> %ptr, %value, %comparator: !spirv.ptr<i32, Workgroup>
  return %0: i32
}

// CHECK-LABEL: subgroup_ballot
func.func @subgroup_ballot(%predicate: i1) -> vector<4xi32> {
  // CHECK: min version: v1.3
  // CHECK: max version: v1.6
  // CHECK: extensions: [ ]
  // CHECK: capabilities: [ [GroupNonUniformBallot] ]
  %0 = spirv.GroupNonUniformBallot <Workgroup> %predicate : vector<4xi32>
  return %0: vector<4xi32>
}

// CHECK-LABEL: module_logical_glsl450
func.func @module_logical_glsl450() {
  // CHECK: spirv.module min version: v1.0
  // CHECK: spirv.module max version: v1.6
  // CHECK: spirv.module extensions: [ ]
  // CHECK: spirv.module capabilities: [ [Shader] ]
  spirv.module Logical GLSL450 { }
  return
}

// CHECK-LABEL: module_physical_storage_buffer64_vulkan
func.func @module_physical_storage_buffer64_vulkan() {
  // CHECK: spirv.module min version: v1.0
  // CHECK: spirv.module max version: v1.6
  // CHECK: spirv.module extensions: [ [SPV_EXT_physical_storage_buffer, SPV_KHR_physical_storage_buffer] [SPV_KHR_vulkan_memory_model] ]
  // CHECK: spirv.module capabilities: [ [PhysicalStorageBufferAddresses] [VulkanMemoryModel] ]
  spirv.module PhysicalStorageBuffer64 Vulkan { }
  return
}

//===----------------------------------------------------------------------===//
// Integer Dot Product ops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: sdot_scalar_i32_i32
func.func @sdot_scalar_i32_i32(%a: i32) -> i32 {
  // CHECK: min version: v1.0
  // CHECK: max version: v1.6
  // CHECK: extensions: [ [SPV_KHR_integer_dot_product] ]
  // CHECK: capabilities: [ [DotProduct] [DotProductInput4x8BitPacked] ]
  %r = spirv.SDot %a, %a, <PackedVectorFormat4x8Bit>: i32 -> i32
  return %r: i32
}

// CHECK-LABEL: sdot_vector_4xi8_i64
func.func @sdot_vector_4xi8_i64(%a: vector<4xi8>) -> i64 {
  // CHECK: min version: v1.0
  // CHECK: max version: v1.6
  // CHECK: extensions: [ [SPV_KHR_integer_dot_product] ]
  // CHECK: capabilities: [ [DotProduct] [DotProductInput4x8Bit] ]
  %r = spirv.SDot %a, %a: vector<4xi8> -> i64
  return %r: i64
}

// CHECK-LABEL: sdot_vector_4xi16_i64
func.func @sdot_vector_4xi16_i64(%a: vector<4xi16>) -> i64 {
  // CHECK: min version: v1.0
  // CHECK: max version: v1.6
  // CHECK: extensions: [ [SPV_KHR_integer_dot_product] ]
  // CHECK: capabilities: [ [DotProduct] [DotProductInputAll] ]
  %r = spirv.SDot %a, %a: vector<4xi16> -> i64
  return %r: i64
}

// CHECK-LABEL: sudot_scalar_i32_i32
func.func @sudot_scalar_i32_i32(%a: i32) -> i32 {
  // CHECK: min version: v1.0
  // CHECK: max version: v1.6
  // CHECK: extensions: [ [SPV_KHR_integer_dot_product] ]
  // CHECK: capabilities: [ [DotProduct] [DotProductInput4x8BitPacked] ]
  %r = spirv.SUDot %a, %a, <PackedVectorFormat4x8Bit>: i32 -> i32
  return %r: i32
}

// CHECK-LABEL: sudot_vector_4xi8_i64
func.func @sudot_vector_4xi8_i64(%a: vector<4xi8>) -> i64 {
  // CHECK: min version: v1.0
  // CHECK: max version: v1.6
  // CHECK: extensions: [ [SPV_KHR_integer_dot_product] ]
  // CHECK: capabilities: [ [DotProduct] [DotProductInput4x8Bit] ]
  %r = spirv.SUDot %a, %a: vector<4xi8> -> i64
  return %r: i64
}

// CHECK-LABEL: sudot_vector_4xi16_i64
func.func @sudot_vector_4xi16_i64(%a: vector<4xi16>) -> i64 {
  // CHECK: min version: v1.0
  // CHECK: max version: v1.6
  // CHECK: extensions: [ [SPV_KHR_integer_dot_product] ]
  // CHECK: capabilities: [ [DotProduct] [DotProductInputAll] ]
  %r = spirv.SUDot %a, %a: vector<4xi16> -> i64
  return %r: i64
}

// CHECK-LABEL: udot_scalar_i32_i32
func.func @udot_scalar_i32_i32(%a: i32) -> i32 {
  // CHECK: min version: v1.0
  // CHECK: max version: v1.6
  // CHECK: extensions: [ [SPV_KHR_integer_dot_product] ]
  // CHECK: capabilities: [ [DotProduct] [DotProductInput4x8BitPacked] ]
  %r = spirv.UDot %a, %a, <PackedVectorFormat4x8Bit>: i32 -> i32
  return %r: i32
}

// CHECK-LABEL: udot_vector_4xi8_i64
func.func @udot_vector_4xi8_i64(%a: vector<4xi8>) -> i64 {
  // CHECK: min version: v1.0
  // CHECK: max version: v1.6
  // CHECK: extensions: [ [SPV_KHR_integer_dot_product] ]
  // CHECK: capabilities: [ [DotProduct] [DotProductInput4x8Bit] ]
  %r = spirv.UDot %a, %a: vector<4xi8> -> i64
  return %r: i64
}

// CHECK-LABEL: udot_vector_4xi16_i64
func.func @udot_vector_4xi16_i64(%a: vector<4xi16>) -> i64 {
  // CHECK: min version: v1.0
  // CHECK: max version: v1.6
  // CHECK: extensions: [ [SPV_KHR_integer_dot_product] ]
  // CHECK: capabilities: [ [DotProduct] [DotProductInputAll] ]
  %r = spirv.UDot %a, %a: vector<4xi16> -> i64
  return %r: i64
}

// CHECK-LABEL: sdot_acc_sat_scalar_i32_i32
func.func @sdot_acc_sat_scalar_i32_i32(%a: i32) -> i32 {
  // CHECK: min version: v1.0
  // CHECK: max version: v1.6
  // CHECK: extensions: [ [SPV_KHR_integer_dot_product] ]
  // CHECK: capabilities: [ [DotProduct] [DotProductInput4x8BitPacked] ]
  %r = spirv.SDotAccSat %a, %a, %a, <PackedVectorFormat4x8Bit>: i32 -> i32
  return %r: i32
}

// CHECK-LABEL: sdot_acc_sat_vector_4xi8_i64
func.func @sdot_acc_sat_vector_4xi8_i64(%a: vector<4xi8>, %acc: i64) -> i64 {
  // CHECK: min version: v1.0
  // CHECK: max version: v1.6
  // CHECK: extensions: [ [SPV_KHR_integer_dot_product] ]
  // CHECK: capabilities: [ [DotProduct] [DotProductInput4x8Bit] ]
  %r = spirv.SDotAccSat %a, %a, %acc: vector<4xi8> -> i64
  return %r: i64
}

// CHECK-LABEL: sdot_acc_sat_vector_4xi16_i64
func.func @sdot_acc_sat_vector_4xi16_i64(%a: vector<4xi16>, %acc: i64) -> i64 {
  // CHECK: min version: v1.0
  // CHECK: max version: v1.6
  // CHECK: extensions: [ [SPV_KHR_integer_dot_product] ]
  // CHECK: capabilities: [ [DotProduct] [DotProductInputAll] ]
  %r = spirv.SDotAccSat %a, %a, %acc: vector<4xi16> -> i64
  return %r: i64
}

// CHECK-LABEL: sudot_acc_sat_scalar_i32_i32
func.func @sudot_acc_sat_scalar_i32_i32(%a: i32) -> i32 {
  // CHECK: min version: v1.0
  // CHECK: max version: v1.6
  // CHECK: extensions: [ [SPV_KHR_integer_dot_product] ]
  // CHECK: capabilities: [ [DotProduct] [DotProductInput4x8BitPacked] ]
  %r = spirv.SUDotAccSat %a, %a, %a, <PackedVectorFormat4x8Bit>: i32 -> i32
  return %r: i32
}

// CHECK-LABEL: sudot_acc_sat_vector_4xi8_i64
func.func @sudot_acc_sat_vector_4xi8_i64(%a: vector<4xi8>, %acc: i64) -> i64 {
  // CHECK: min version: v1.0
  // CHECK: max version: v1.6
  // CHECK: extensions: [ [SPV_KHR_integer_dot_product] ]
  // CHECK: capabilities: [ [DotProduct] [DotProductInput4x8Bit] ]
  %r = spirv.SUDotAccSat %a, %a, %acc: vector<4xi8> -> i64
  return %r: i64
}

// CHECK-LABEL: sudot_acc_sat_vector_4xi16_i64
func.func @sudot_acc_sat_vector_4xi16_i64(%a: vector<4xi16>, %acc: i64) -> i64 {
  // CHECK: min version: v1.0
  // CHECK: max version: v1.6
  // CHECK: extensions: [ [SPV_KHR_integer_dot_product] ]
  // CHECK: capabilities: [ [DotProduct] [DotProductInputAll] ]
  %r = spirv.SUDotAccSat %a, %a, %acc: vector<4xi16> -> i64
  return %r: i64
}

// CHECK-LABEL: udot_acc_sat_scalar_i32_i32
func.func @udot_acc_sat_scalar_i32_i32(%a: i32) -> i32 {
  // CHECK: min version: v1.0
  // CHECK: max version: v1.6
  // CHECK: extensions: [ [SPV_KHR_integer_dot_product] ]
  // CHECK: capabilities: [ [DotProduct] [DotProductInput4x8BitPacked] ]
  %r = spirv.UDotAccSat %a, %a, %a, <PackedVectorFormat4x8Bit>: i32 -> i32
  return %r: i32
}

// CHECK-LABEL: udot_acc_sat_vector_4xi8_i64
func.func @udot_acc_sat_vector_4xi8_i64(%a: vector<4xi8>, %acc: i64) -> i64 {
  // CHECK: min version: v1.0
  // CHECK: max version: v1.6
  // CHECK: extensions: [ [SPV_KHR_integer_dot_product] ]
  // CHECK: capabilities: [ [DotProduct] [DotProductInput4x8Bit] ]
  %r = spirv.UDotAccSat %a, %a, %acc: vector<4xi8> -> i64
  return %r: i64
}

// CHECK-LABEL: udot_acc_sat_vector_4xi16_i64
func.func @udot_acc_sat_vector_4xi16_i64(%a: vector<4xi16>, %acc: i64) -> i64 {
  // CHECK: min version: v1.0
  // CHECK: max version: v1.6
  // CHECK: extensions: [ [SPV_KHR_integer_dot_product] ]
  // CHECK: capabilities: [ [DotProduct] [DotProductInputAll] ]
  %r = spirv.UDotAccSat %a, %a, %acc: vector<4xi16> -> i64
  return %r: i64
}

//===----------------------------------------------------------------------===//
// Primitive ops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: emit_vertex
func.func @emit_vertex() -> () {
  // CHECK: min version: v1.0
  // CHECK: max version: v1.6
  // CHECK: capabilities: [ [Geometry] ]
  spirv.EmitVertex
  return
}

// CHECK-LABEL: end_primitive
func.func @end_primitive() -> () {
  // CHECK: min version: v1.0
  // CHECK: max version: v1.6
  // CHECK: capabilities: [ [Geometry] ]
  spirv.EndPrimitive
  return
}

//===----------------------------------------------------------------------===//
// Mesh ops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: emit_mesh_tasks
func.func @emit_mesh_tasks(%0 : i32) -> () {
  // CHECK: min version: v1.4
  // CHECK: max version: v1.6
  // CHECK: extensions: [ [SPV_EXT_mesh_shader] ]
  // CHECK: capabilities: [ [MeshShadingEXT] ]
  spirv.EXT.EmitMeshTasks %0, %0, %0 : i32, i32, i32
}

// CHECK-LABEL: set_mesh_outputs
func.func @set_mesh_outputs(%0 : i32, %1 : i32) -> () {
  // CHECK: min version: v1.4
  // CHECK: max version: v1.6
  // CHECK: extensions: [ [SPV_EXT_mesh_shader] ]
  // CHECK: capabilities: [ [MeshShadingEXT] ]
  spirv.EXT.SetMeshOutputs %0, %1 : i32, i32
  spirv.Return
}
