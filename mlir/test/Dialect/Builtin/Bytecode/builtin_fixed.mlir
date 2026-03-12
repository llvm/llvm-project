// RUN: mlir-opt -allow-unregistered-dialect -emit-bytecode -emit-bytecode-producer=fixed %s -o %t
// Verify unchanged.
// RUN: cmp -s %t %p/builtin_fixed_0.mlirbc
// Verify can read as expected.
// RUN: mlir-opt -allow-unregistered-dialect -mlir-print-local-scope %t | FileCheck %s

// Regression test for all builtin dialect bytecode attribute and
// type encodings. Exercises every entry in BuiltinDialectAttributes and
// BuiltinDialectTypes including special cases (splats, empty containers,
// new float types, ranges, etc.) [to be confirmed].

// allow-unregisterd-dialect is set to allow for the string constant types.

module {

//===----------------------------------------------------------------------===//
// ArrayAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestArrayAttr
module @TestArrayAttr attributes {
  // CHECK-DAG: bytecode.empty = []
  // CHECK-DAG: bytecode.single = [unit]
  // CHECK-DAG: bytecode.nested = {{\[}}[1 : i32, 2 : i32], [3 : i32]]
  bytecode.empty = [],
  bytecode.single = [unit],
  bytecode.nested = [[1 : i32, 2 : i32], [3 : i32]]
} {} loc(unknown)

//===----------------------------------------------------------------------===//
// DictionaryAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestDictionaryAttr
module @TestDictionaryAttr attributes {
  // CHECK-DAG: bytecode.empty_dict = {}
  // CHECK-DAG: bytecode.nested_dict = {inner = {a = 1 : i32}}
  bytecode.empty_dict = {},
  bytecode.nested_dict = {inner = {a = 1 : i32}}
} {} loc(unknown)

//===----------------------------------------------------------------------===//
// StringAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestStringAttr
module @TestStringAttr attributes {
  // CHECK-DAG: bytecode.plain = "hello world"
  // CHECK-DAG: bytecode.empty_str = ""
  // CHECK-DAG: bytecode.typed = "typed" : i32
  bytecode.plain = "hello world",
  bytecode.empty_str = "",
  bytecode.typed = "typed" : i32
} {} loc(unknown)

//===----------------------------------------------------------------------===//
// FlatSymbolRefAttr / SymbolRefAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestSymbolRefAttr
module @TestSymbolRefAttr attributes {
  // CHECK-DAG: bytecode.flat = @flat_sym
  // CHECK-DAG: bytecode.nested = @root::@child1::@child2
  bytecode.flat = @flat_sym,
  bytecode.nested = @root::@child1::@child2
} {} loc(unknown)

//===----------------------------------------------------------------------===//
// TypeAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestTypeAttr
module @TestTypeAttr attributes {
  // CHECK-DAG: bytecode.type_i32 = i32
  // CHECK-DAG: bytecode.type_memref = memref<2x3xf32>
  bytecode.type_i32 = i32,
  bytecode.type_memref = memref<2x3xf32>
} {} loc(unknown)

//===----------------------------------------------------------------------===//
// UnitAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestUnitAttr
module @TestUnitAttr attributes {
  // CHECK-DAG: bytecode.unit
  bytecode.unit
} {} loc(unknown)

//===----------------------------------------------------------------------===//
// IntegerAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestIntegerAttr
module @TestIntegerAttr attributes {
  // CHECK-DAG: bytecode.bool_false = false
  // CHECK-DAG: bytecode.bool_true = true
  // CHECK-DAG: bytecode.i8_neg = -1 : i8
  // CHECK-DAG: bytecode.i32_val = 42 : i32
  // CHECK-DAG: bytecode.si32_val = -100 : si32
  // CHECK-DAG: bytecode.ui64_val = 800 : ui64
  // CHECK-DAG: bytecode.i128_large = 90000000000000000300000000000000000001 : i128
  // CHECK-DAG: bytecode.index_val = 7 : index
  bytecode.bool_false = false,
  bytecode.bool_true = true,
  bytecode.i8_neg = -1 : i8,
  bytecode.i32_val = 42 : i32,
  bytecode.si32_val = -100 : si32,
  bytecode.ui64_val = 800 : ui64,
  bytecode.i128_large = 90000000000000000300000000000000000001 : i128,
  bytecode.index_val = 7 : index
} {} loc(unknown)

//===----------------------------------------------------------------------===//
// FloatAttr — all float types
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestFloatAttr
module @TestFloatAttr attributes {
  // CHECK-DAG: bytecode.bf16 = -5.000000e-01 : bf16
  // CHECK-DAG: bytecode.f16 = 1.500000e+00 : f16
  // CHECK-DAG: bytecode.f32 = 3.140000e+00 : f32
  // CHECK-DAG: bytecode.f64 = 1.000000e+01 : f64
  // CHECK-DAG: bytecode.f80 = 0.1{{.*}} : f80
  // CHECK-DAG: bytecode.f128 = 0.1{{.*}} : f128
  bytecode.bf16 = -0.5 : bf16,
  bytecode.f16 = 1.5 : f16,
  bytecode.f32 = 3.14 : f32,
  bytecode.f64 = 10.0 : f64,
  bytecode.f80 = 0.1 : f80,
  bytecode.f128 = 0.1 : f128
} {} loc(unknown)

//===----------------------------------------------------------------------===//
// DenseArrayAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestDenseArrayAttr
module @TestDenseArrayAttr attributes {
  // CHECK-DAG: bytecode.bool_arr = array<i1: true, false, true>
  // CHECK-DAG: bytecode.i8_arr = array<i8: 10, 32, -1>
  // CHECK-DAG: bytecode.i16_arr = array<i16: 100, -200>
  // CHECK-DAG: bytecode.i32_arr = array<i32: 1, 2, 3>
  // CHECK-DAG: bytecode.i64_arr = array<i64: 1000000, -1000000>
  // CHECK-DAG: bytecode.f32_arr = array<f32: 1.000000e+00, 2.000000e+00>
  // CHECK-DAG: bytecode.f64_arr = array<f64: 3.140000e+00>
  // CHECK-DAG: bytecode.empty_arr = array<i32>
  bytecode.bool_arr = array<i1: true, false, true>,
  bytecode.i8_arr = array<i8: 10, 32, 255>,
  bytecode.i16_arr = array<i16: 100, -200>,
  bytecode.i32_arr = array<i32: 1, 2, 3>,
  bytecode.i64_arr = array<i64: 1000000, -1000000>,
  bytecode.f32_arr = array<f32: 1.0, 2.0>,
  bytecode.f64_arr = array<f64: 3.14>,
  bytecode.empty_arr = array<i32>
} {} loc(unknown)

//===----------------------------------------------------------------------===//
// DenseIntOrFPElementsAttr — including splats
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestDenseIntOrFPElementsAttr
module @TestDenseIntOrFPElementsAttr attributes {
  // Splats
  // CHECK-DAG: bytecode.splat_i1 = dense<true> : tensor<256xi1>
  // CHECK-DAG: bytecode.splat_i32 = dense<42> : tensor<4x4xi32>
  // CHECK-DAG: bytecode.splat_f32 = dense<1.000000e+00> : tensor<2x3xf32>
  // CHECK-DAG: bytecode.splat_f64 = dense<0.000000e+00> : tensor<8xf64>
  // Non-splat
  // CHECK-DAG: bytecode.dense_i1 = dense<[true, false, true]> : tensor<3xi1>
  // CHECK-DAG: bytecode.dense_i8 = dense<[10, 20, 30]> : tensor<3xi8>
  // CHECK-DAG: bytecode.dense_f32 = dense<[1.{{.*}}, 2.{{.*}}]> : tensor<2xf32>
  // Multi-dimensional
  // CHECK-DAG: bytecode.dense_2d = dense<{{\[}}[1, 2], [3, 4]]> : tensor<2x2xi32>
  bytecode.splat_i1 = dense<true> : tensor<256xi1>,
  bytecode.splat_i32 = dense<42> : tensor<4x4xi32>,
  bytecode.splat_f32 = dense<1.0> : tensor<2x3xf32>,
  bytecode.splat_f64 = dense<0.0> : tensor<8xf64>,
  bytecode.dense_i1 = dense<[true, false, true]> : tensor<3xi1>,
  bytecode.dense_i8 = dense<[10, 20, 30]> : tensor<3xi8>,
  bytecode.dense_f32 = dense<[1.0, 2.0]> : tensor<2xf32>,
  bytecode.dense_2d = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
} {} loc(unknown)

//===----------------------------------------------------------------------===//
// SparseElementsAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestSparseElementsAttr
module @TestSparseElementsAttr attributes {
  // CHECK-LITERAL: bytecode.sparse = sparse<[[0, 0], [1, 2]], [1, 5]> : tensor<3x4xi32>
  // CHECK-LITERAL: bytecode.sparse_1d = sparse<[1, 3], [10.0, 20.0]> : tensor<5xf32>
  bytecode.sparse = sparse<[[0, 0], [1, 2]], [1, 5]> : tensor<3x4xi32>,
  bytecode.sparse_1d = sparse<[1, 3], [10.0, 20.0]> : tensor<5xf32>
} {} loc(unknown)

//===----------------------------------------------------------------------===//
// DistinctAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestDistinctAttr
module @TestDistinctAttr attributes {
  // CHECK-DAG: bytecode.diff = distinct[0]<42 : i32>
  // CHECK-DAG: bytecode.same1 = distinct[1]<42 : i32>
  // CHECK-DAG: bytecode.same2 = distinct[1]<42 : i32>
  bytecode.same1 = distinct[0]<42 : i32>,
  bytecode.same2 = distinct[0]<42 : i32>,
  bytecode.diff = distinct[1]<42 : i32>
} {} loc(unknown)

//===----------------------------------------------------------------------===//
// Location Attributes
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestLocationAttrs
module @TestLocationAttrs attributes {
  // CallSiteLoc
  // CHECK-DAG: bytecode.callsite = loc(callsite("callee" at "caller.cc":5:3))
  bytecode.callsite = loc(callsite("callee" at "caller.cc":5:3)),
  // FileLineColLoc
  // CHECK-DAG: bytecode.flc = loc("source.cc":10:8)
  bytecode.flc = loc("source.cc":10:8),
  // FileLineColRange
  // CHECK-DAG: bytecode.flc_range1 = loc("source.cc":10:8 to 12:4)
  // CHECK-DAG: bytecode.flc_range2 = loc("source.cc":10:8 to :12)
  // CHECK-DAG: bytecode.flc_range3 = loc("source.cc":10:8 to 12:8)
  bytecode.flc_range1 = loc("source.cc":10:8 to 12:4),
  bytecode.flc_range2 = loc("source.cc":10:8 to :12),
  bytecode.flc_range3 = loc("source.cc":10:8 to 12:8),
  // FusedLoc (without metadata)
  // CHECK-DAG: bytecode.fused = loc(fused["a", "b":1:2])
  bytecode.fused = loc(fused["a", "b":1:2]),
  // FusedLoc (with metadata)
  // CHECK-DAG: bytecode.fused_meta = loc(fused<"myPass">["x", "y"])
  bytecode.fused_meta = loc(fused<"myPass">["x", "y"]),
  // NameLoc (without child)
  // CHECK-DAG: bytecode.name = loc("named")
  bytecode.name = loc("named"),
  // NameLoc (with child)
  // CHECK-DAG: bytecode.name_child = loc("named"("child.cc":1:1))
  bytecode.name_child = loc("named"("child.cc":1:1)),
  // UnknownLoc
  // CHECK-DAG: bytecode.unknown = loc(unknown)
  bytecode.unknown = loc(unknown)
} {} loc(unknown)

//===----------------------------------------------------------------------===//
// All Float Types (type-level roundtrip)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestAllFloatTypes
module @TestAllFloatTypes attributes {
  // CHECK-DAG: bytecode.bf16 = bf16
  // CHECK-DAG: bytecode.f16 = f16
  // CHECK-DAG: bytecode.f32 = f32
  // CHECK-DAG: bytecode.f64 = f64
  // CHECK-DAG: bytecode.f80 = f80
  // CHECK-DAG: bytecode.f128 = f128
  bytecode.bf16 = bf16,
  bytecode.f16 = f16,
  bytecode.f32 = f32,
  bytecode.f64 = f64,
  bytecode.f80 = f80,
  bytecode.f128 = f128
} {} loc(unknown)

//===----------------------------------------------------------------------===//
// IntegerType, IndexType, NoneType
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestBasicTypes
module @TestBasicTypes attributes {
  // CHECK-DAG: bytecode.i1 = i1
  // CHECK-DAG: bytecode.i8 = i8
  // CHECK-DAG: bytecode.i32 = i32
  // CHECK-DAG: bytecode.i1024 = i1024
  // CHECK-DAG: bytecode.si32 = si32
  // CHECK-DAG: bytecode.ui64 = ui64
  // CHECK-DAG: bytecode.index = index
  // CHECK-DAG: bytecode.none = none
  bytecode.i1 = i1,
  bytecode.i8 = i8,
  bytecode.i32 = i32,
  bytecode.i1024 = i1024,
  bytecode.si32 = si32,
  bytecode.ui64 = ui64,
  bytecode.index = index,
  bytecode.none = none
} {} loc(unknown)

//===----------------------------------------------------------------------===//
// FunctionType
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestFunctionType
module @TestFunctionType attributes {
  // CHECK-DAG: bytecode.empty_func = () -> ()
  // CHECK-DAG: bytecode.func_args = (i32, f64) -> i1
  // CHECK-DAG: bytecode.func_multi_res = (i32) -> (f32, f64)
  bytecode.empty_func = () -> (),
  bytecode.func_args = (i32, f64) -> (i1),
  bytecode.func_multi_res = (i32) -> (f32, f64)
} {} loc(unknown)

//===----------------------------------------------------------------------===//
// ComplexType
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestComplexType
module @TestComplexType attributes {
  // CHECK-DAG: bytecode.c_i32 = complex<i32>
  // CHECK-DAG: bytecode.c_f64 = complex<f64>
  bytecode.c_i32 = complex<i32>,
  bytecode.c_f64 = complex<f64>
} {} loc(unknown)

//===----------------------------------------------------------------------===//
// MemRefType (with/without memory space, layout)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestMemRefType
module @TestMemRefType attributes {
  // CHECK-DAG: bytecode.simple = memref<2x3xf32>
  // CHECK-DAG: bytecode.with_memspace = memref<4xi8, 1>
  // CHECK-DAG: bytecode.dynamic = memref<?x?xf32>
  bytecode.simple = memref<2x3xf32>,
  bytecode.with_memspace = memref<4xi8, 1>,
  bytecode.dynamic = memref<?x?xf32>
} {} loc(unknown)

//===----------------------------------------------------------------------===//
// RankedTensorType (with/without encoding, dynamic dims)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestRankedTensorType
module @TestRankedTensorType attributes {
  // CHECK-DAG: bytecode.static = tensor<16x32xf64>
  // CHECK-DAG: bytecode.dynamic = tensor<?x32x?xf32>
  // CHECK-DAG: bytecode.with_encoding = tensor<16xf64, "sparse">
  bytecode.static = tensor<16x32xf64>,
  bytecode.dynamic = tensor<?x32x?xf32>,
  bytecode.with_encoding = tensor<16xf64, "sparse">
} {} loc(unknown)

//===----------------------------------------------------------------------===//
// UnrankedTensorType
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestUnrankedTensorType
module @TestUnrankedTensorType attributes {
  // CHECK-DAG: bytecode.unranked = tensor<*xi8>
  bytecode.unranked = tensor<*xi8>
} {} loc(unknown)

//===----------------------------------------------------------------------===//
// UnrankedMemRefType (with/without memory space)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestUnrankedMemRefType
module @TestUnrankedMemRefType attributes {
  // CHECK-DAG: bytecode.plain = memref<*xi8>
  // CHECK-DAG: bytecode.with_memspace = memref<*xi8, 1>
  bytecode.plain = memref<*xi8>,
  bytecode.with_memspace = memref<*xi8, 1>
} {} loc(unknown)

//===----------------------------------------------------------------------===//
// VectorType (with/without scalable dims)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestVectorType
module @TestVectorType attributes {
  // CHECK-DAG: bytecode.fixed = vector<8x8x128xi8>
  // CHECK-DAG: bytecode.scalable = vector<8x[8]xf32>
  // CHECK-DAG: bytecode.all_scalable = vector<[4]x[4]xf16>
  bytecode.fixed = vector<8x8x128xi8>,
  bytecode.scalable = vector<8x[8]xf32>,
  bytecode.all_scalable = vector<[4]x[4]xf16>
} {} loc(unknown)

//===----------------------------------------------------------------------===//
// TupleType
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestTupleType
module @TestTupleType attributes {
  // CHECK-DAG: bytecode.empty_tuple = tuple<>
  // CHECK-DAG: bytecode.mixed_tuple = tuple<i32, f64, index>
  bytecode.empty_tuple = tuple<>,
  bytecode.mixed_tuple = tuple<i32, f64, index>
} {} loc(unknown)

//===----------------------------------------------------------------------===//
// DenseResourceElementsAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestDenseResourceElementsAttr
module @TestDenseResourceElementsAttr attributes {
  // CHECK-DAG: bytecode.resource = dense_resource<blob1> : tensor<3xi64>
  bytecode.resource = dense_resource<blob1> : tensor<3xi64>
} {} loc(unknown)

//===----------------------------------------------------------------------===//
// DenseStringElementsAttr — splat and non-splat
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestDenseStringElementsAttr
module @TestDenseStringElementsAttr attributes {
  bytecode.splat_str = dense<"splat"> : tensor<4x!bytecode.string>,
  bytecode.dense_str = dense<["foo", "bar", "baz"]> : tensor<3x!bytecode.string>
} {} loc(unknown)

} loc(unknown)

{-#
  dialect_resources: {
    builtin: {
      blob1: "0x08000000010000000000000002000000000000000300000000000000"
    }
  }
#-}

