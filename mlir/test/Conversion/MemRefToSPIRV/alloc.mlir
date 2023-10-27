// RUN: mlir-opt -split-input-file -convert-memref-to-spirv -canonicalize -verify-diagnostics %s -o - | FileCheck %s

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
  }
{
  func.func @alloc_dealloc_workgroup_mem_shader_f32(%arg0 : index, %arg1 : index) {
    %0 = memref.alloc() : memref<4x5xf32, #spirv.storage_class<Workgroup>>
    %1 = memref.load %0[%arg0, %arg1] : memref<4x5xf32, #spirv.storage_class<Workgroup>>
    memref.store %1, %0[%arg0, %arg1] : memref<4x5xf32, #spirv.storage_class<Workgroup>>
    memref.dealloc %0 : memref<4x5xf32, #spirv.storage_class<Workgroup>>
    return
  }
}
//       CHECK: spirv.GlobalVariable @[[$VAR:.+]] : !spirv.ptr<!spirv.struct<(!spirv.array<20 x f32>)>, Workgroup>
// CHECK-LABEL: func @alloc_dealloc_workgroup_mem_shader_f32
//   CHECK-NOT:   memref.alloc
//       CHECK:   %[[PTR:.+]] = spirv.mlir.addressof @[[$VAR]]
//       CHECK:   %[[LOADPTR:.+]] = spirv.AccessChain %[[PTR]]
//       CHECK:   %[[VAL:.+]] = spirv.Load "Workgroup" %[[LOADPTR]] : f32
//       CHECK:   %[[STOREPTR:.+]] = spirv.AccessChain %[[PTR]]
//       CHECK:   spirv.Store "Workgroup" %[[STOREPTR]], %[[VAL]] : f32
//   CHECK-NOT:   memref.dealloc

// -----

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
  }
{
  func.func @alloc_dealloc_workgroup_mem_shader_i16(%arg0 : index, %arg1 : index) {
    %0 = memref.alloc() : memref<4x5xi16, #spirv.storage_class<Workgroup>>
    %1 = memref.load %0[%arg0, %arg1] : memref<4x5xi16, #spirv.storage_class<Workgroup>>
    memref.store %1, %0[%arg0, %arg1] : memref<4x5xi16, #spirv.storage_class<Workgroup>>
    memref.dealloc %0 : memref<4x5xi16, #spirv.storage_class<Workgroup>>
    return
  }
}

//       CHECK: spirv.GlobalVariable @__workgroup_mem__{{[0-9]+}}
//  CHECK-SAME:   !spirv.ptr<!spirv.struct<(!spirv.array<10 x i32>)>, Workgroup>
// CHECK-LABEL: func @alloc_dealloc_workgroup_mem_shader_i16
//       CHECK:   %[[VAR:.+]] = spirv.mlir.addressof @__workgroup_mem__0
//       CHECK:   %[[LOC:.+]] = spirv.SDiv
//       CHECK:   %[[PTR:.+]] = spirv.AccessChain %[[VAR]][%{{.+}}, %[[LOC]]]
//       CHECK:   %{{.+}} = spirv.Load "Workgroup" %[[PTR]] : i32
//       CHECK:   %[[LOC:.+]] = spirv.SDiv
//       CHECK:   %[[PTR:.+]] = spirv.AccessChain %[[VAR]][%{{.+}}, %[[LOC]]]
//       CHECK:   %{{.+}} = spirv.AtomicAnd "Workgroup" "AcquireRelease" %[[PTR]], %{{.+}} : !spirv.ptr<i32, Workgroup>
//       CHECK:   %{{.+}} = spirv.AtomicOr "Workgroup" "AcquireRelease" %[[PTR]], %{{.+}} : !spirv.ptr<i32, Workgroup>

// -----

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
  }
{
  func.func @two_allocs() {
    %0 = memref.alloc() : memref<4x5xf32, #spirv.storage_class<Workgroup>>
    %1 = memref.alloc() : memref<2x3xi32, #spirv.storage_class<Workgroup>>
    return
  }
}

//   CHECK-DAG: spirv.GlobalVariable @__workgroup_mem__{{[0-9]+}}
//  CHECK-SAME:   !spirv.ptr<!spirv.struct<(!spirv.array<6 x i32>)>, Workgroup>
//   CHECK-DAG: spirv.GlobalVariable @__workgroup_mem__{{[0-9]+}}
//  CHECK-SAME:   !spirv.ptr<!spirv.struct<(!spirv.array<20 x f32>)>, Workgroup>
// CHECK-LABEL: func @two_allocs()

// -----

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
  }
{
  func.func @two_allocs_vector() {
    %0 = memref.alloc() : memref<4xvector<4xf32>, #spirv.storage_class<Workgroup>>
    %1 = memref.alloc() : memref<2xvector<2xi32>, #spirv.storage_class<Workgroup>>
    return
  }
}

//   CHECK-DAG: spirv.GlobalVariable @__workgroup_mem__{{[0-9]+}}
//  CHECK-SAME:   !spirv.ptr<!spirv.struct<(!spirv.array<2 x vector<2xi32>>)>, Workgroup>
//   CHECK-DAG: spirv.GlobalVariable @__workgroup_mem__{{[0-9]+}}
//  CHECK-SAME:   !spirv.ptr<!spirv.struct<(!spirv.array<4 x vector<4xf32>>)>, Workgroup>
// CHECK-LABEL: func @two_allocs_vector()

// -----

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
  }
{
  // CHECK-LABEL: func @alloc_dynamic_size
  func.func @alloc_dynamic_size(%arg0 : index) -> f32 {
    // CHECK: memref.alloc
    %0 = memref.alloc(%arg0) : memref<4x?xf32, #spirv.storage_class<Workgroup>>
    %1 = memref.load %0[%arg0, %arg0] : memref<4x?xf32, #spirv.storage_class<Workgroup>>
    return %1: f32
  }
}

// -----

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
  }
{
  // CHECK-LABEL: func @alloc_unsupported_memory_space
  func.func @alloc_unsupported_memory_space(%arg0: index) -> f32 {
    // CHECK: memref.alloc
    %0 = memref.alloc() : memref<4x5xf32, #spirv.storage_class<StorageBuffer>>
    %1 = memref.load %0[%arg0, %arg0] : memref<4x5xf32, #spirv.storage_class<StorageBuffer>>
    return %1: f32
  }
}


// -----

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
  }
{
  // CHECK-LABEL: func @dealloc_dynamic_size
  func.func @dealloc_dynamic_size(%arg0 : memref<4x?xf32, #spirv.storage_class<Workgroup>>) {
    // CHECK: memref.dealloc
    memref.dealloc %arg0 : memref<4x?xf32, #spirv.storage_class<Workgroup>>
    return
  }
}

// -----

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
  }
{
  // CHECK-LABEL: func @dealloc_unsupported_memory_space
  func.func @dealloc_unsupported_memory_space(%arg0 : memref<4x5xf32, #spirv.storage_class<StorageBuffer>>) {
    // CHECK: memref.dealloc
    memref.dealloc %arg0 : memref<4x5xf32, #spirv.storage_class<StorageBuffer>>
    return
  }
}

// -----
module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Kernel], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
  }
{
  func.func @alloc_dealloc_workgroup_mem_kernel(%arg0 : index, %arg1 : index) {
    %0 = memref.alloc() : memref<4x5xf32, #spirv.storage_class<Workgroup>>
    %1 = memref.load %0[%arg0, %arg1] : memref<4x5xf32, #spirv.storage_class<Workgroup>>
    memref.store %1, %0[%arg0, %arg1] : memref<4x5xf32, #spirv.storage_class<Workgroup>>
    memref.dealloc %0 : memref<4x5xf32, #spirv.storage_class<Workgroup>>
    return
  }
}
//       CHECK: spirv.GlobalVariable @[[$VAR:.+]] : !spirv.ptr<!spirv.array<20 x f32>, Workgroup>
// CHECK-LABEL: func @alloc_dealloc_workgroup_mem_kernel
//   CHECK-NOT:   memref.alloc
//       CHECK:   %[[PTR:.+]] = spirv.mlir.addressof @[[$VAR]]
//       CHECK:   %[[LOADPTR:.+]] = spirv.AccessChain %[[PTR]]
//       CHECK:   %[[VAL:.+]] = spirv.Load "Workgroup" %[[LOADPTR]] : f32
//       CHECK:   %[[STOREPTR:.+]] = spirv.AccessChain %[[PTR]]
//       CHECK:   spirv.Store "Workgroup" %[[STOREPTR]], %[[VAL]] : f32
//   CHECK-NOT:   memref.dealloc

// -----

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
  }
{
  func.func @zero_size() {
    %0 = memref.alloc() : memref<0xf32, #spirv.storage_class<Workgroup>>
    return
  }
}

// Zero-sized allocations are not handled yet. Just make sure we do not crash.
// CHECK-LABEL: func @zero_size()
