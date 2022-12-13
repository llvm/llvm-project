// RUN: mlir-opt -emit-bytecode -allow-unregistered-dialect %s | mlir-opt -allow-unregistered-dialect -mlir-print-local-scope | FileCheck %s

// Bytecode currently does not support big-endian platforms
// UNSUPPORTED: target=s390x-{{.*}}

//===----------------------------------------------------------------------===//
// ArrayAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestArray
module @TestArray attributes {
  // CHECK: bytecode.array = [unit]
  bytecode.array = [unit]
} {}

//===----------------------------------------------------------------------===//
// DenseArrayAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestDenseArray
module @TestDenseArray attributes {
  // CHECK: bytecode.test1 = array<i1: true, false, true, false, false>
  // CHECK: bytecode.test2 = array<i8: 10, 32, -1>
  // CHECK: bytecode.test3 = array<f64: 1.{{.*}}e+01, 3.2{{.*}}e+01, 1.809{{.*}}e+03
  bytecode.test1 = array<i1: true, false, true, false, false>,
  bytecode.test2 = array<i8: 10, 32, 255>,
  bytecode.test3 = array<f64: 10.0, 32.0, 1809.0>
} {}

//===----------------------------------------------------------------------===//
// DenseIntOfFPElementsAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestDenseIntOrFPElements
// CHECK: bytecode.test1 = dense<true> : tensor<256xi1>
// CHECK: bytecode.test2 = dense<[10, 32, -1]> : tensor<3xi8>
// CHECK: bytecode.test3 = dense<[1.{{.*}}e+01, 3.2{{.*}}e+01, 1.809{{.*}}e+03]> : tensor<3xf64>
module @TestDenseIntOrFPElements attributes {
  bytecode.test1 = dense<true> : tensor<256xi1>,
  bytecode.test2 = dense<[10, 32, 255]> : tensor<3xi8>,
  bytecode.test3 = dense<[10.0, 32.0, 1809.0]> : tensor<3xf64>
} {}

//===----------------------------------------------------------------------===//
// DenseStringElementsAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestDenseStringElementsAttr
module @TestDenseStringElementsAttr attributes {
  bytecode.test1 = dense<"splat"> : tensor<256x!bytecode.string>,
  bytecode.test2 = dense<["foo", "bar", "baz"]> : tensor<3x!bytecode.string>
} {}

//===----------------------------------------------------------------------===//
// FloatAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestFloat
module @TestFloat attributes {
  // CHECK: bytecode.float = 1.000000e+01 : f64
  // CHECK: bytecode.float1 = 0.10000{{.*}} : f80
  // CHECK: bytecode.float2 = 0.10000{{.*}} : f128
  // CHECK: bytecode.float3 = -5.000000e-01 : bf16
  bytecode.float = 10.0 : f64,
  bytecode.float1 = 0.1 : f80,
  bytecode.float2 = 0.1 : f128,
  bytecode.float3 = -0.5 : bf16
} {}

//===----------------------------------------------------------------------===//
// IntegerAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestInt
module @TestInt attributes {
  // CHECK: bytecode.int = false
  // CHECK: bytecode.int1 = -1 : i8
  // CHECK: bytecode.int2 = 800 : ui64
  // CHECK: bytecode.int3 = 90000000000000000300000000000000000001 : i128
  bytecode.int = false,
  bytecode.int1 = -1 : i8,
  bytecode.int2 = 800 : ui64,
  bytecode.int3 = 90000000000000000300000000000000000001 : i128
} {}

//===----------------------------------------------------------------------===//
// SparseElementsAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestSparseElements
module @TestSparseElements attributes {
  // CHECK-LITERAL: bytecode.sparse = sparse<[[0, 0], [1, 2]], [1, 5]> : tensor<3x4xi32>
  bytecode.sparse = sparse<[[0, 0], [1, 2]], [1, 5]> : tensor<3x4xi32>
} {}


//===----------------------------------------------------------------------===//
// StringAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestString
module @TestString attributes {
  // CHECK: bytecode.string = "hello"
  // CHECK: bytecode.string2 = "hello" : i32
  bytecode.string = "hello",
  bytecode.string2 = "hello" : i32
} {}

//===----------------------------------------------------------------------===//
// SymbolRefAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestSymbolRef
module @TestSymbolRef attributes {
  // CHECK: bytecode.ref = @foo
  // CHECK: bytecode.ref2 = @foo::@bar::@foo
  bytecode.ref = @foo,
  bytecode.ref2 = @foo::@bar::@foo
} {}

//===----------------------------------------------------------------------===//
// TypeAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestType
module @TestType attributes {
  // CHECK: bytecode.type = i178
  bytecode.type = i178
} {}

//===----------------------------------------------------------------------===//
// CallSiteLoc
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestLocCallSite
module @TestLocCallSite attributes {
  // CHECK: bytecode.loc = loc(callsite("foo" at "mysource.cc":10:8))
  bytecode.loc = loc(callsite("foo" at "mysource.cc":10:8))
} {}

//===----------------------------------------------------------------------===//
// FileLineColLoc
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestLocFileLineCol
module @TestLocFileLineCol attributes {
  // CHECK: bytecode.loc = loc("mysource.cc":10:8)
  bytecode.loc = loc("mysource.cc":10:8)
} {}

//===----------------------------------------------------------------------===//
// FusedLoc
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestLocFused
module @TestLocFused attributes {
  // CHECK: bytecode.loc = loc(fused["foo", "mysource.cc":10:8])
  // CHECK: bytecode.loc2 = loc(fused<"myPass">["foo", "foo2"])
  bytecode.loc = loc(fused["foo", "mysource.cc":10:8]),
  bytecode.loc2 = loc(fused<"myPass">["foo", "foo2"])
} {}

//===----------------------------------------------------------------------===//
// NameLoc
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestLocName
module @TestLocName attributes {
  // CHECK: bytecode.loc = loc("foo")
  // CHECK: bytecode.loc2 = loc("foo"("mysource.cc":10:8))
  bytecode.loc = loc("foo"),
  bytecode.loc2 = loc("foo"("mysource.cc":10:8))
} {}

//===----------------------------------------------------------------------===//
// UnknownLoc
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestLocUnknown
module @TestLocUnknown attributes {
  // CHECK: bytecode.loc = loc(unknown)
  bytecode.loc = loc(unknown)
} {}
