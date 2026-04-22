// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK: memref<?x?xf32, strided<[?, ?]>>
func.func private @f1() -> memref<?x?xf32, strided<[?, ?]>>
// CHECK: memref<?x?xf32, strided<[42, 1]>>
func.func private @f2() -> memref<?x?xf32, strided<[42, 1]>>
// CHECK: memref<?x?xf32, strided<[?, 1]>>
func.func private @f3() -> memref<?x?xf32, strided<[?, 1]>>
// CHECK: memref<?x?xf32, strided<[?, 1]>>
func.func private @f4() -> memref<?x?xf32, strided<[?, 1]>>
// CHECK: memref<?x?xf32, strided<[42, 1]>>
func.func private @f5() -> memref<?x?xf32, strided<[42, 1]>>
// CHECK: memref<?x?xf32, strided<[42, 1]>>
func.func private @f6() -> memref<?x?xf32, strided<[42, 1]>>
// CHECK: memref<f32, strided<[]>>
func.func private @f7() -> memref<f32, strided<[]>>
// CHECK: memref<f32, strided<[]>>
func.func private @f8() -> memref<f32, strided<[]>>
// CHECK: memref<?xf32, strided<[-1]>>
func.func private @f9() -> memref<?xf32, strided<[-1]>>
// CHECK: memref<f32, strided<[]>>
func.func private @f10() -> memref<f32, strided<[]>>
