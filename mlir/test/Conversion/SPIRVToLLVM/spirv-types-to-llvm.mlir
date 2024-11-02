// RUN: mlir-opt -split-input-file -convert-spirv-to-llvm -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Array type
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @array(!llvm.array<16 x f32>, !llvm.array<32 x vector<4xf32>>)
spirv.func @array(!spirv.array<16 x f32>, !spirv.array< 32 x vector<4xf32> >) "None"

// CHECK-LABEL: @array_with_natural_stride(!llvm.array<16 x f32>)
spirv.func @array_with_natural_stride(!spirv.array<16 x f32, stride=4>) "None"

//===----------------------------------------------------------------------===//
// Pointer type
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @pointer_scalar(!llvm.ptr<i1>, !llvm.ptr<f32>)
spirv.func @pointer_scalar(!spirv.ptr<i1, Uniform>, !spirv.ptr<f32, Private>) "None"

// CHECK-LABEL: @pointer_vector(!llvm.ptr<vector<4xi32>>)
spirv.func @pointer_vector(!spirv.ptr<vector<4xi32>, Function>) "None"

//===----------------------------------------------------------------------===//
// Runtime array type
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @runtime_array_vector(!llvm.array<0 x vector<4xf32>>)
spirv.func @runtime_array_vector(!spirv.rtarray< vector<4xf32> >) "None"

// CHECK-LABEL: @runtime_array_scalar(!llvm.array<0 x f32>)
spirv.func @runtime_array_scalar(!spirv.rtarray<f32>) "None"

//===----------------------------------------------------------------------===//
// Struct type
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @struct(!llvm.struct<packed (f64)>)
spirv.func @struct(!spirv.struct<(f64)>) "None"

// CHECK-LABEL: @struct_nested(!llvm.struct<packed (i32, struct<packed (i64, i32)>)>)
spirv.func @struct_nested(!spirv.struct<(i32, !spirv.struct<(i64, i32)>)>) "None"

// CHECK-LABEL: @struct_with_natural_offset(!llvm.struct<(i8, i32)>)
spirv.func @struct_with_natural_offset(!spirv.struct<(i8[0], i32[4])>) "None"
