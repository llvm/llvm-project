// RUN: mlir-opt -emit-bytecode %s | mlir-opt | FileCheck %s

// Bytecode currently does not support big-endian platforms
// UNSUPPORTED: s390x-

//===----------------------------------------------------------------------===//
// ComplexType
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestComplex
module @TestComplex attributes {
  // CHECK: bytecode.test = complex<i32>
  bytecode.test = complex<i32>
} {}

//===----------------------------------------------------------------------===//
// FloatType
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestFloat
module @TestFloat attributes {
  // CHECK: bytecode.test = bf16,
  // CHECK: bytecode.test1 = f16,
  // CHECK: bytecode.test2 = f32,
  // CHECK: bytecode.test3 = f64,
  // CHECK: bytecode.test4 = f80,
  // CHECK: bytecode.test5 = f128
  bytecode.test = bf16,
  bytecode.test1 = f16,
  bytecode.test2 = f32,
  bytecode.test3 = f64,
  bytecode.test4 = f80,
  bytecode.test5 = f128
} {}

//===----------------------------------------------------------------------===//
// IntegerType
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestInteger
module @TestInteger attributes {
  // CHECK: bytecode.int = i1024,
  // CHECK: bytecode.int1 = si32,
  // CHECK: bytecode.int2 = ui512
  bytecode.int = i1024,
  bytecode.int1 = si32,
  bytecode.int2 = ui512
} {}

//===----------------------------------------------------------------------===//
// IndexType
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestIndex
module @TestIndex attributes {
  // CHECK: bytecode.index = index
  bytecode.index = index
} {}

//===----------------------------------------------------------------------===//
// FunctionType
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestFunc
module @TestFunc attributes {
  // CHECK: bytecode.func = () -> (),
  // CHECK: bytecode.func1 = (i1) -> i32
  bytecode.func = () -> (),
  bytecode.func1 = (i1) -> (i32)
} {}

//===----------------------------------------------------------------------===//
// MemRefType
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestMemRef
module @TestMemRef attributes {
  // CHECK: bytecode.test = memref<2xi8>,
  // CHECK: bytecode.test1 = memref<2xi8, 1>
  bytecode.test = memref<2xi8>,
  bytecode.test1 = memref<2xi8, 1>
} {}

//===----------------------------------------------------------------------===//
// NoneType
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestNone
module @TestNone attributes {
  // CHECK: bytecode.test = none
  bytecode.test = none
} {}

//===----------------------------------------------------------------------===//
// RankedTensorType
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestRankedTensor
module @TestRankedTensor attributes {
  // CHECK: bytecode.test = tensor<16x32x?xf64>,
  // CHECK: bytecode.test1 = tensor<16xf64, "sparse">
  bytecode.test = tensor<16x32x?xf64>,
  bytecode.test1 = tensor<16xf64, "sparse">
} {}

//===----------------------------------------------------------------------===//
// TupleType
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestTuple
module @TestTuple attributes {
  // CHECK: bytecode.test = tuple<>,
  // CHECK: bytecode.test1 = tuple<i32, i1, f32>
  bytecode.test = tuple<>,
  bytecode.test1 = tuple<i32, i1, f32>
} {}

//===----------------------------------------------------------------------===//
// UnrankedMemRefType
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestUnrankedMemRef
module @TestUnrankedMemRef attributes {
  // CHECK: bytecode.test = memref<*xi8>,
  // CHECK: bytecode.test1 = memref<*xi8, 1>
  bytecode.test = memref<*xi8>,
  bytecode.test1 = memref<*xi8, 1>
} {}

//===----------------------------------------------------------------------===//
// UnrankedTensorType
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestUnrankedTensor
module @TestUnrankedTensor attributes {
  // CHECK: bytecode.test = tensor<*xi8>
  bytecode.test = tensor<*xi8>
} {}

//===----------------------------------------------------------------------===//
// VectorType
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestVector
module @TestVector attributes {
  // CHECK: bytecode.test = vector<8x8x128xi8>,
  // CHECK: bytecode.test1 = vector<8x[8]xf32>
  bytecode.test = vector<8x8x128xi8>,
  bytecode.test1 = vector<8x[8]xf32>
} {}
