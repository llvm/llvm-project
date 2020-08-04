// RUN: mlir-opt -split-input-file -convert-spirv-to-llvm -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Array type
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @array(!llvm.array<16 x float>, !llvm.array<32 x vec<4 x float>>)
func @array(!spv.array<16xf32>, !spv.array< 32 x vector<4xf32> >) -> ()

//===----------------------------------------------------------------------===//
// Pointer type
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @pointer_scalar(!llvm.ptr<i1>, !llvm.ptr<float>)
func @pointer_scalar(!spv.ptr<i1, Uniform>, !spv.ptr<f32, Private>) -> ()

// CHECK-LABEL: @pointer_vector(!llvm.ptr<vec<4 x i32>>)
func @pointer_vector(!spv.ptr<vector<4xi32>, Function>) -> ()

//===----------------------------------------------------------------------===//
// Runtime array type
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @runtime_array_vector(!llvm.array<0 x vec<4 x float>>)
func @runtime_array_vector(!spv.rtarray< vector<4xf32> >) -> ()

// CHECK-LABEL: @runtime_array_scalar(!llvm.array<0 x float>)
func @runtime_array_scalar(!spv.rtarray<f32>) -> ()

//===----------------------------------------------------------------------===//
// Struct type
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @struct(!llvm.struct<packed (double)>)
func @struct(!spv.struct<f64>) -> ()

// CHECK-LABEL: @struct_nested(!llvm.struct<packed (i32, struct<packed (i64, i32)>)>)
func @struct_nested(!spv.struct<i32, !spv.struct<i64, i32>>)
