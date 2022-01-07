// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.AccessChain
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @access_chain
spv.func @access_chain() "None" {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
  %0 = spv.Constant 1: i32
  %1 = spv.Variable : !spv.ptr<!spv.struct<(f32, !spv.array<4xf32>)>, Function>
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: llvm.getelementptr %{{.*}}[%[[ZERO]], 1, %[[ONE]]] : (!llvm.ptr<struct<packed (f32, array<4 x f32>)>>, i32, i32) -> !llvm.ptr<f32>
  %2 = spv.AccessChain %1[%0, %0] : !spv.ptr<!spv.struct<(f32, !spv.array<4xf32>)>, Function>, i32, i32
  spv.Return
}

// CHECK-LABEL: @access_chain_array
spv.func @access_chain_array(%arg0 : i32) "None" {
  %0 = spv.Variable : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: llvm.getelementptr %{{.*}}[%[[ZERO]], %{{.*}}] : (!llvm.ptr<array<4 x array<4 x f32>>>, i32, i32) -> !llvm.ptr<array<4 x f32>>
  %1 = spv.AccessChain %0[%arg0] : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>, i32
  %2 = spv.Load "Function" %1 ["Volatile"] : !spv.array<4xf32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.GlobalVariable and spv.mlir.addressof
//===----------------------------------------------------------------------===//

spv.module Logical GLSL450 {
  // CHECK: llvm.mlir.global external constant @var() : f32
  spv.GlobalVariable @var : !spv.ptr<f32, Input>
}

spv.module Logical GLSL450 {
  //       CHECK: llvm.mlir.global private @struct() : !llvm.struct<packed (f32, array<10 x f32>)>
  // CHECK-LABEL: @func
  //       CHECK:   llvm.mlir.addressof @struct : !llvm.ptr<struct<packed (f32, array<10 x f32>)>>
  spv.GlobalVariable @struct : !spv.ptr<!spv.struct<(f32, !spv.array<10xf32>)>, Private>
  spv.func @func() "None" {
    %0 = spv.mlir.addressof @struct : !spv.ptr<!spv.struct<(f32, !spv.array<10xf32>)>, Private>
    spv.Return
  }
}

spv.module Logical GLSL450 {
  //       CHECK: llvm.mlir.global external @bar_descriptor_set0_binding0() : i32
  // CHECK-LABEL: @foo
  //       CHECK:   llvm.mlir.addressof @bar_descriptor_set0_binding0 : !llvm.ptr<i32>
  spv.GlobalVariable @bar bind(0, 0) : !spv.ptr<i32, StorageBuffer>
  spv.func @foo() "None" {
    %0 = spv.mlir.addressof @bar : !spv.ptr<i32, StorageBuffer>
    spv.Return
  }
}

spv.module @name Logical GLSL450 {
  //       CHECK: llvm.mlir.global external @name_bar_descriptor_set0_binding0() : i32
  // CHECK-LABEL: @foo
  //       CHECK:   llvm.mlir.addressof @name_bar_descriptor_set0_binding0 : !llvm.ptr<i32>
  spv.GlobalVariable @bar bind(0, 0) : !spv.ptr<i32, StorageBuffer>
  spv.func @foo() "None" {
    %0 = spv.mlir.addressof @bar : !spv.ptr<i32, StorageBuffer>
    spv.Return
  }
}

spv.module Logical GLSL450 {
  // CHECK: llvm.mlir.global external @bar() {location = 1 : i32} : i32
  // CHECK-LABEL: @foo
  spv.GlobalVariable @bar {location = 1 : i32} : !spv.ptr<i32, Output>
  spv.func @foo() "None" {
    %0 = spv.mlir.addressof @bar : !spv.ptr<i32, Output>
    spv.Return
  }
}

spv.module Logical GLSL450 {
  // CHECK: llvm.mlir.global external constant @bar() {location = 3 : i32} : f32
  // CHECK-LABEL: @foo
  spv.GlobalVariable @bar {descriptor_set = 0 : i32, location = 3 : i32} : !spv.ptr<f32, UniformConstant>
  spv.func @foo() "None" {
    %0 = spv.mlir.addressof @bar : !spv.ptr<f32, UniformConstant>
    spv.Return
  }
}

//===----------------------------------------------------------------------===//
// spv.Load
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @load
spv.func @load() "None" {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  //  CHECK: llvm.load %{{.*}} : !llvm.ptr<f32>
  %1 = spv.Load "Function" %0 : f32
  spv.Return
}

// CHECK-LABEL: @load_none
spv.func @load_none() "None" {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  //  CHECK: llvm.load %{{.*}} : !llvm.ptr<f32>
  %1 = spv.Load "Function" %0 ["None"] : f32
  spv.Return
}

// CHECK-LABEL: @load_with_alignment
spv.func @load_with_alignment() "None" {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // CHECK: llvm.load %{{.*}} {alignment = 4 : i64} : !llvm.ptr<f32>
  %1 = spv.Load "Function" %0 ["Aligned", 4] : f32
  spv.Return
}

// CHECK-LABEL: @load_volatile
spv.func @load_volatile() "None" {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // CHECK: llvm.load volatile %{{.*}} : !llvm.ptr<f32>
  %1 = spv.Load "Function" %0 ["Volatile"] : f32
  spv.Return
}

// CHECK-LABEL: @load_nontemporal
spv.func @load_nontemporal() "None" {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // CHECK: llvm.load %{{.*}} {nontemporal} : !llvm.ptr<f32>
  %1 = spv.Load "Function" %0 ["Nontemporal"] : f32
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.Store
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @store
spv.func @store(%arg0 : f32) "None" {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // CHECK: llvm.store %{{.*}}, %{{.*}} : !llvm.ptr<f32>
  spv.Store "Function" %0, %arg0 : f32
  spv.Return
}

// CHECK-LABEL: @store_composite
spv.func @store_composite(%arg0 : !spv.struct<(f64)>) "None" {
  %0 = spv.Variable : !spv.ptr<!spv.struct<(f64)>, Function>
  // CHECK: llvm.store %{{.*}}, %{{.*}} : !llvm.ptr<struct<packed (f64)>>
  spv.Store "Function" %0, %arg0 : !spv.struct<(f64)>
  spv.Return
}

// CHECK-LABEL: @store_with_alignment
spv.func @store_with_alignment(%arg0 : f32) "None" {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 4 : i64} : !llvm.ptr<f32>
  spv.Store "Function" %0, %arg0 ["Aligned", 4] : f32
  spv.Return
}

// CHECK-LABEL: @store_volatile
spv.func @store_volatile(%arg0 : f32) "None" {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // CHECK: llvm.store volatile %{{.*}}, %{{.*}} : !llvm.ptr<f32>
  spv.Store "Function" %0, %arg0 ["Volatile"] : f32
  spv.Return
}

// CHECK-LABEL: @store_nontemporal
spv.func @store_nontemporal(%arg0 : f32) "None" {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // CHECK: llvm.store %{{.*}}, %{{.*}} {nontemporal} : !llvm.ptr<f32>
  spv.Store "Function" %0, %arg0 ["Nontemporal"] : f32
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.Variable
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @variable_scalar
spv.func @variable_scalar() "None" {
  // CHECK: %[[SIZE1:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: llvm.alloca %[[SIZE1]] x f32 : (i32) -> !llvm.ptr<f32>
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // CHECK: %[[SIZE2:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: llvm.alloca %[[SIZE2]] x i8 : (i32) -> !llvm.ptr<i8>
  %1 = spv.Variable : !spv.ptr<i8, Function>
  spv.Return
}

// CHECK-LABEL: @variable_scalar_with_initialization
spv.func @variable_scalar_with_initialization() "None" {
  // CHECK: %[[VALUE:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCATED:.*]] = llvm.alloca %[[SIZE]] x i64 : (i32) -> !llvm.ptr<i64>
  // CHECK: llvm.store %[[VALUE]], %[[ALLOCATED]] : !llvm.ptr<i64>
  %c = spv.Constant 0 : i64
  %0 = spv.Variable init(%c) : !spv.ptr<i64, Function>
  spv.Return
}

// CHECK-LABEL: @variable_vector
spv.func @variable_vector() "None" {
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: llvm.alloca  %[[SIZE]] x vector<3xf32> : (i32) -> !llvm.ptr<vector<3xf32>>
  %0 = spv.Variable : !spv.ptr<vector<3xf32>, Function>
  spv.Return
}

// CHECK-LABEL: @variable_vector_with_initialization
spv.func @variable_vector_with_initialization() "None" {
  // CHECK: %[[VALUE:.*]] = llvm.mlir.constant(dense<false> : vector<3xi1>) : vector<3xi1>
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCATED:.*]] = llvm.alloca %[[SIZE]] x vector<3xi1> : (i32) -> !llvm.ptr<vector<3xi1>>
  // CHECK: llvm.store %[[VALUE]], %[[ALLOCATED]] : !llvm.ptr<vector<3xi1>>
  %c = spv.Constant dense<false> : vector<3xi1>
  %0 = spv.Variable init(%c) : !spv.ptr<vector<3xi1>, Function>
  spv.Return
}

// CHECK-LABEL: @variable_array
spv.func @variable_array() "None" {
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: llvm.alloca %[[SIZE]] x !llvm.array<10 x i32> : (i32) -> !llvm.ptr<array<10 x i32>>
  %0 = spv.Variable : !spv.ptr<!spv.array<10 x i32>, Function>
  spv.Return
}
