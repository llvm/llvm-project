// RUN: mlir-opt -emit-bytecode %s | mlir-opt -mlir-print-local-scope | FileCheck %s

// Bytecode currently does not support big-endian platforms
// UNSUPPORTED: s390x-

//===----------------------------------------------------------------------===//
// ArrayAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestArray
module @TestArray attributes {
  // CHECK: bytecode.array = [unit]
  bytecode.array = [unit]
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
