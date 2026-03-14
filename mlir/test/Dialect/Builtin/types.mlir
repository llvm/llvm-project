// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK: memref<?x?xf32, strided<[?, ?], offset: ?>>
func.func private @f1() -> memref<?x?xf32, strided<[?, ?], offset: ?>>
// CHECK: memref<?x?xf32, strided<[42, 1], offset: 10>>
func.func private @f2() -> memref<?x?xf32, strided<[42, 1], offset: 10>>
// CHECK: memref<?x?xf32, strided<[?, 1], offset: 10>>
func.func private @f3() -> memref<?x?xf32, strided<[?, 1], offset: 10>>
// CHECK: memref<?x?xf32, strided<[?, 1], offset: ?>>
func.func private @f4() -> memref<?x?xf32, strided<[?, 1], offset: ?>>
// CHECK: memref<?x?xf32, strided<[42, 1]>>
func.func private @f5() -> memref<?x?xf32, strided<[42, 1]>>
// CHECK: memref<?x?xf32, strided<[42, 1]>>
func.func private @f6() -> memref<?x?xf32, strided<[42, 1], offset: 0>>
// CHECK: memref<f32, strided<[]>>
func.func private @f7() -> memref<f32, strided<[]>>
// CHECK: memref<f32, strided<[], offset: ?>>
func.func private @f8() -> memref<f32, strided<[], offset: ?>>
// CHECK: memref<?xf32, strided<[-1], offset: ?>>
func.func private @f9() -> memref<?xf32, strided<[-1], offset: ?>>
// CHECK: memref<f32, strided<[], offset: -1>>
func.func private @f10() -> memref<f32, strided<[], offset: -1>>
