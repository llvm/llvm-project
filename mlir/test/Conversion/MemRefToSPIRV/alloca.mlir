// RUN: mlir-opt -split-input-file -convert-memref-to-spirv -canonicalize -verify-diagnostics %s -o - | FileCheck %s

module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], []>, {}>} {
  func.func @alloc_function_variable(%arg0 : index, %arg1 : index) {
    %0 = memref.alloca() : memref<4x5xf32, 6>
    %1 = memref.load %0[%arg0, %arg1] : memref<4x5xf32, 6>
    memref.store %1, %0[%arg0, %arg1] : memref<4x5xf32, 6>
    return
  }
}

// CHECK-LABEL: func @alloc_function_variable
//       CHECK:   %[[VAR:.+]] = spv.Variable : !spv.ptr<!spv.struct<(!spv.array<20 x f32, stride=4>)>, Function>
//       CHECK:   %[[LOADPTR:.+]] = spv.AccessChain %[[VAR]]
//       CHECK:   %[[VAL:.+]] = spv.Load "Function" %[[LOADPTR]] : f32
//       CHECK:   %[[STOREPTR:.+]] = spv.AccessChain %[[VAR]]
//       CHECK:   spv.Store "Function" %[[STOREPTR]], %[[VAL]] : f32


// -----

module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], []>, {}>} {
  func.func @two_allocs() {
    %0 = memref.alloca() : memref<4x5xf32, 6>
    %1 = memref.alloca() : memref<2x3xi32, 6>
    return
  }
}

// CHECK-LABEL: func @two_allocs
//   CHECK-DAG: spv.Variable : !spv.ptr<!spv.struct<(!spv.array<6 x i32, stride=4>)>, Function>
//   CHECK-DAG: spv.Variable : !spv.ptr<!spv.struct<(!spv.array<20 x f32, stride=4>)>, Function>

// -----

module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], []>, {}>} {
  func.func @two_allocs_vector() {
    %0 = memref.alloca() : memref<4xvector<4xf32>, 6>
    %1 = memref.alloca() : memref<2xvector<2xi32>, 6>
    return
  }
}

// CHECK-LABEL: func @two_allocs_vector
//   CHECK-DAG: spv.Variable : !spv.ptr<!spv.struct<(!spv.array<2 x vector<2xi32>, stride=8>)>, Function>
//   CHECK-DAG: spv.Variable : !spv.ptr<!spv.struct<(!spv.array<4 x vector<4xf32>, stride=16>)>, Function>


// -----

module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], []>, {}>} {
  // CHECK-LABEL: func @alloc_dynamic_size
  func.func @alloc_dynamic_size(%arg0 : index) -> f32 {
    // CHECK: memref.alloca
    %0 = memref.alloca(%arg0) : memref<4x?xf32, 6>
    %1 = memref.load %0[%arg0, %arg0] : memref<4x?xf32, 6>
    return %1: f32
  }
}

// -----

module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], []>, {}>} {
  // CHECK-LABEL: func @alloc_unsupported_memory_space
  func.func @alloc_unsupported_memory_space(%arg0: index) -> f32 {
    // CHECK: memref.alloca
    %0 = memref.alloca() : memref<4x5xf32>
    %1 = memref.load %0[%arg0, %arg0] : memref<4x5xf32>
    return %1: f32
  }
}
