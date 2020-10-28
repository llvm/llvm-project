// RUN: mlir-opt -split-input-file -convert-spirv-to-llvm -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Array type
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @array(!llvm.array<16 x float>, !llvm.array<32 x vec<4 x float>>)
spv.func @array(!spv.array<16 x f32>, !spv.array< 32 x vector<4xf32> >) "None"

// CHECK-LABEL: @array_with_natural_stride(!llvm.array<16 x float>)
spv.func @array_with_natural_stride(!spv.array<16 x f32, stride=4>) "None"

//===----------------------------------------------------------------------===//
// Pointer type
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @pointer_scalar(!llvm.ptr<i1>, !llvm.ptr<float>)
spv.func @pointer_scalar(!spv.ptr<i1, Uniform>, !spv.ptr<f32, Private>) "None"

// CHECK-LABEL: @pointer_vector(!llvm.ptr<vec<4 x i32>>)
spv.func @pointer_vector(!spv.ptr<vector<4xi32>, Function>) "None"

//===----------------------------------------------------------------------===//
// Runtime array type
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @runtime_array_vector(!llvm.array<0 x vec<4 x float>>)
spv.func @runtime_array_vector(!spv.rtarray< vector<4xf32> >) "None"

// CHECK-LABEL: @runtime_array_scalar(!llvm.array<0 x float>)
spv.func @runtime_array_scalar(!spv.rtarray<f32>) "None"

//===----------------------------------------------------------------------===//
// Struct type
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @struct(!llvm.struct<packed (double)>)
spv.func @struct(!spv.struct<f64>) "None"

// CHECK-LABEL: @struct_nested(!llvm.struct<packed (i32, struct<packed (i64, i32)>)>)
spv.func @struct_nested(!spv.struct<i32, !spv.struct<i64, i32>>) "None"

// CHECK-LABEL: @struct_with_natural_offset(!llvm.struct<(i8, i32)>)
spv.func @struct_with_natural_offset(!spv.struct<i8[0], i32[4]>) "None"
