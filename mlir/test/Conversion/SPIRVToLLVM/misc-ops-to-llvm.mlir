// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.CompositeExtract
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @composite_extract_array
spv.func @composite_extract_array(%arg: !spv.array<4x!spv.array<4xf32>>) "None" {
  // CHECK: llvm.extractvalue %{{.*}}[1 : i32, 3 : i32] : !llvm.array<4 x array<4 x float>>
  %0 = spv.CompositeExtract %arg[1 : i32, 3 : i32] : !spv.array<4x!spv.array<4xf32>>
  spv.Return
}

// CHECK-LABEL: @composite_extract_vector
spv.func @composite_extract_vector(%arg: vector<3xf32>) "None" {
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
  // CHECK: llvm.extractelement %{{.*}}[%[[ZERO]] : !llvm.i32] : !llvm.vec<3 x float>
  %0 = spv.CompositeExtract %arg[0 : i32] : vector<3xf32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.CompositeInsert
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @composite_insert_struct
spv.func @composite_insert_struct(%arg0: i32, %arg1: !spv.struct<f32, !spv.array<4xi32>>) "None" {
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[1 : i32, 3 : i32] : !llvm.struct<packed (float, array<4 x i32>)>
  %0 = spv.CompositeInsert %arg0, %arg1[1 : i32, 3 : i32] : i32 into !spv.struct<f32, !spv.array<4xi32>>
  spv.Return
}

// CHECK-LABEL: @composite_insert_vector
spv.func @composite_insert_vector(%arg0: vector<3xf32>, %arg1: f32) "None" {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
  // CHECK: llvm.insertelement %{{.*}}, %{{.*}}[%[[ONE]] : !llvm.i32] : !llvm.vec<3 x float>
  %0 = spv.CompositeInsert %arg1, %arg0[1 : i32] : f32 into vector<3xf32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.Select
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @select_scalar
spv.func @select_scalar(%arg0: i1, %arg1: vector<3xi32>, %arg2: f32) "None" {
  // CHECK: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : !llvm.i1, !llvm.vec<3 x i32>
  %0 = spv.Select %arg0, %arg1, %arg1 : i1, vector<3xi32>
  // CHECK: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : !llvm.i1, !llvm.float
  %1 = spv.Select %arg0, %arg2, %arg2 : i1, f32
  spv.Return
}

// CHECK-LABEL: @select_vector
spv.func @select_vector(%arg0: vector<2xi1>, %arg1: vector<2xi32>) "None" {
  // CHECK: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : !llvm.vec<2 x i1>, !llvm.vec<2 x i32>
  %0 = spv.Select %arg0, %arg1, %arg1 : vector<2xi1>, vector<2xi32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.EntryPoint and spv.ExecutionMode
//===----------------------------------------------------------------------===//

//      CHECK: module {
// CHECK-NEXT:   llvm.func @empty
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
spv.module Logical GLSL450 {
  spv.func @empty() "None" {
    spv.Return
  }
  spv.EntryPoint "GLCompute" @empty
  spv.ExecutionMode @empty "LocalSize", 1, 1, 1
}

//===----------------------------------------------------------------------===//
// spv.Undef
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @undef_scalar
spv.func @undef_scalar() "None" {
  // CHECK: llvm.mlir.undef : !llvm.float
  %0 = spv.undef : f32
  spv.Return
}

// CHECK-LABEL: @undef_vector
spv.func @undef_vector() "None" {
  // CHECK: llvm.mlir.undef : !llvm.vec<2 x i32>
  %0 = spv.undef : vector<2xi32>
  spv.Return
}
