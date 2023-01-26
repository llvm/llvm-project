// RUN: mlir-opt -split-input-file -convert-memref-to-spirv -canonicalize -verify-diagnostics %s -o - | FileCheck %s

module attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader], []>, #spirv.resource_limits<>>} {
  func.func @alloc_function_variable(%arg0 : index, %arg1 : index) {
    %0 = memref.alloca() : memref<4x5xf32, #spirv.storage_class<Function>>
    %1 = memref.load %0[%arg0, %arg1] : memref<4x5xf32, #spirv.storage_class<Function>>
    memref.store %1, %0[%arg0, %arg1] : memref<4x5xf32, #spirv.storage_class<Function>>
    return
  }
}

// CHECK-LABEL: func @alloc_function_variable
//       CHECK:   %[[VAR:.+]] = spirv.Variable : !spirv.ptr<!spirv.struct<(!spirv.array<20 x f32>)>, Function>
//       CHECK:   %[[LOADPTR:.+]] = spirv.AccessChain %[[VAR]]
//       CHECK:   %[[VAL:.+]] = spirv.Load "Function" %[[LOADPTR]] : f32
//       CHECK:   %[[STOREPTR:.+]] = spirv.AccessChain %[[VAR]]
//       CHECK:   spirv.Store "Function" %[[STOREPTR]], %[[VAL]] : f32


// -----

module attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader], []>, #spirv.resource_limits<>>} {
  func.func @two_allocs() {
    %0 = memref.alloca() : memref<4x5xf32, #spirv.storage_class<Function>>
    %1 = memref.alloca() : memref<2x3xi32, #spirv.storage_class<Function>>
    return
  }
}

// CHECK-LABEL: func @two_allocs
//   CHECK-DAG: spirv.Variable : !spirv.ptr<!spirv.struct<(!spirv.array<6 x i32>)>, Function>
//   CHECK-DAG: spirv.Variable : !spirv.ptr<!spirv.struct<(!spirv.array<20 x f32>)>, Function>

// -----

module attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader], []>, #spirv.resource_limits<>>} {
  func.func @two_allocs_vector() {
    %0 = memref.alloca() : memref<4xvector<4xf32>, #spirv.storage_class<Function>>
    %1 = memref.alloca() : memref<2xvector<2xi32>, #spirv.storage_class<Function>>
    return
  }
}

// CHECK-LABEL: func @two_allocs_vector
//   CHECK-DAG: spirv.Variable : !spirv.ptr<!spirv.struct<(!spirv.array<2 x vector<2xi32>>)>, Function>
//   CHECK-DAG: spirv.Variable : !spirv.ptr<!spirv.struct<(!spirv.array<4 x vector<4xf32>>)>, Function>


// -----

module attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader], []>, #spirv.resource_limits<>>} {
  // CHECK-LABEL: func @alloc_dynamic_size
  func.func @alloc_dynamic_size(%arg0 : index) -> f32 {
    // CHECK: memref.alloca
    %0 = memref.alloca(%arg0) : memref<4x?xf32, #spirv.storage_class<Function>>
    %1 = memref.load %0[%arg0, %arg0] : memref<4x?xf32, #spirv.storage_class<Function>>
    return %1: f32
  }
}

// -----

module attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader], []>, #spirv.resource_limits<>>} {
  // CHECK-LABEL: func @alloc_unsupported_memory_space
  func.func @alloc_unsupported_memory_space(%arg0: index) -> f32 {
    // CHECK: memref.alloca
    %0 = memref.alloca() : memref<4x5xf32>
    %1 = memref.load %0[%arg0, %arg0] : memref<4x5xf32>
    return %1: f32
  }
}
