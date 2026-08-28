// RUN: mlir-opt -emit-bytecode -allow-unregistered-dialect %s | mlir-opt -allow-unregistered-dialect -mlir-print-local-scope | FileCheck %s

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
// DenseTypedElementsAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestDenseTypedElements
// CHECK-DAG: bytecode.test1 = dense<true> : tensor<256xi1>
// CHECK-DAG: bytecode.test2 = dense<[10, 32, -1]> : tensor<3xi8>
// CHECK-DAG: bytecode.test3 = dense<[1.{{.*}}e+01, 3.2{{.*}}e+01, 1.809{{.*}}e+03]> : tensor<3xf64>
// CHECK-DAG: bytecode.i16 = dense<[100, -200]> : tensor<2xi16>
// CHECK-DAG: bytecode.i64 = dense<1000000> : tensor<4xi64>
// CHECK-DAG: bytecode.f16 = dense<[1.500000e+00, 2.500000e+00]> : tensor<2xf16>
// CHECK-DAG: bytecode.bf16 = dense<[-5.000000e-01, 5.000000e-01]> : tensor<2xbf16>
// CHECK-DAG: bytecode.complex_val = dense<(1.000000e+00,2.000000e+00)> : tensor<1xcomplex<f32>>
module @TestDenseTypedElements attributes {
  bytecode.test1 = dense<true> : tensor<256xi1>,
  bytecode.test2 = dense<[10, 32, 255]> : tensor<3xi8>,
  bytecode.test3 = dense<[10.0, 32.0, 1809.0]> : tensor<3xf64>,
  bytecode.i16 = dense<[100, -200]> : tensor<2xi16>,
  bytecode.i64 = dense<1000000> : tensor<4xi64>,
  bytecode.f16 = dense<[1.5, 2.5]> : tensor<2xf16>,
  bytecode.bf16 = dense<[-0.5, 0.5]> : tensor<2xbf16>,
  bytecode.complex_val = dense<(1.0, 2.0)> : tensor<1xcomplex<f32>>
} {}

// DenseI1 (single, 8-element, varied sizes)

// CHECK-LABEL: @TestDenseI1SingleElement
module @TestDenseI1SingleElement attributes {
  // CHECK-DAG: bytecode.false_1 = dense<false> : tensor<1xi1>
  // CHECK-DAG: bytecode.true_1 = dense<true> : tensor<1xi1>
  bytecode.false_1 = dense<false> : tensor<1xi1>,
  bytecode.true_1 = dense<true> : tensor<1xi1>
} {}

// CHECK-LABEL: @TestDenseI1EightElements
module @TestDenseI1EightElements attributes {
  // CHECK-DAG: bytecode.mixed_8 = dense<[true, false, true, false, true, false, true, false]> : tensor<8xi1>
  // CHECK-DAG: bytecode.all_true_8 = dense<true> : tensor<8xi1>
  // CHECK-DAG: bytecode.all_false_8 = dense<false> : tensor<8xi1>
  bytecode.mixed_8 = dense<[true, false, true, false, true, false, true, false]> : tensor<8xi1>,
  bytecode.all_true_8 = dense<true> : tensor<8xi1>,
  bytecode.all_false_8 = dense<false> : tensor<8xi1>
} {}

// CHECK-LABEL: @TestDenseI1VariedSizes
module @TestDenseI1VariedSizes attributes {
  // CHECK-DAG: bytecode.i1_2 = dense<[true, false]> : tensor<2xi1>
  // CHECK-DAG: bytecode.i1_7 = dense<[true, false, true, false, true, false, true]> : tensor<7xi1>
  // CHECK-DAG: bytecode.i1_9 = dense<[true, false, true, false, true, false, true, false, true]> : tensor<9xi1>
  // CHECK-DAG: bytecode.i1_16 = dense<[true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false]> : tensor<16xi1>
  bytecode.i1_2 = dense<[true, false]> : tensor<2xi1>,
  bytecode.i1_7 = dense<[true, false, true, false, true, false, true]> : tensor<7xi1>,
  bytecode.i1_9 = dense<[true, false, true, false, true, false, true, false, true]> : tensor<9xi1>,
  bytecode.i1_16 = dense<[true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false]> : tensor<16xi1>
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
  // CHECK: bytecode.inf = 0x7FF0000000000000 : f64
  // CHECK: bytecode.nan = 0x7FF8000000000000 : f64
  // CHECK: bytecode.ninf = 0xFFF0000000000000 : f64
  bytecode.float = 10.0 : f64,
  bytecode.float1 = 0.1 : f80,
  bytecode.float2 = 0.1 : f128,
  bytecode.float3 = -0.5 : bf16,
  bytecode.inf = 0x7FF0000000000000 : f64,
  bytecode.nan = 0x7FF8000000000000 : f64,
  bytecode.ninf = 0xFFF0000000000000 : f64
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
  // CHECK: bytecode.int4 = true
  // CHECK: bytecode.int5 = 42 : i32
  // CHECK: bytecode.int6 = -100 : si32
  // CHECK: bytecode.int7 = 7 : index
  bytecode.int = false,
  bytecode.int1 = -1 : i8,
  bytecode.int2 = 800 : ui64,
  bytecode.int3 = 90000000000000000300000000000000000001 : i128,
  bytecode.int4 = true,
  bytecode.int5 = 42 : i32,
  bytecode.int6 = -100 : si32,
  bytecode.int7 = 7 : index
} {}

//===----------------------------------------------------------------------===//
// SparseElementsAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestSparseElements
module @TestSparseElements attributes {
  // CHECK-LITERAL: bytecode.sparse = sparse<[[0, 0], [1, 2]], [1, 5]> : tensor<3x4xi32>
  // CHECK-LITERAL: bytecode.sparse_1d = sparse<[1, 3], [10.0, 20.0]> : tensor<5xf32>
  bytecode.sparse = sparse<[[0, 0], [1, 2]], [1, 5]> : tensor<3x4xi32>,
  bytecode.sparse_1d = sparse<[1, 3], [10.0, 20.0]> : tensor<5xf32>
} {}


//===----------------------------------------------------------------------===//
// StringAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestString
module @TestString attributes {
  // CHECK: bytecode.string = "hello"
  // CHECK: bytecode.string2 = "hello" : i32
  // CHECK: bytecode.string3 = ""
  bytecode.string = "hello",
  bytecode.string2 = "hello" : i32,
  bytecode.string3 = ""
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
// DistinctAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestDistinct
module @TestDistinct attributes {
  // CHECK: bytecode.distinct = distinct[0]<42 : i32>
  // CHECK: bytecode.distinct2 = distinct[0]<42 : i32>
  // CHECK: bytecode.distinct3 = distinct[1]<42 : i32>
  bytecode.distinct = distinct[0]<42 : i32>,
  bytecode.distinct2 = distinct[0]<42 : i32>,
  bytecode.distinct3 = distinct[1]<42 : i32>
} {}

//===----------------------------------------------------------------------===//
// CallSiteLoc
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestLocCallSite
module @TestLocCallSite attributes {
  // CHECK: bytecode.loc = loc(callsite("foo" at "mysource.cc":10:8))
  // CHECK: bytecode.loc2 = loc(callsite("a":1:1 at callsite("b":2:2 at "c":3:3)))
  bytecode.loc = loc(callsite("foo" at "mysource.cc":10:8)),
  bytecode.loc2 = loc(callsite("a":1:1 at callsite("b":2:2 at "c":3:3)))
} {}

//===----------------------------------------------------------------------===//
// FileLineColLoc
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestLocFileLineCol
module @TestLocFileLineCol attributes {
  // CHECK: bytecode.loc = loc("mysource.cc":10:8)
  // CHECK: bytecode.loc2 = loc("source.cc":10:8 to 12:4)
  // CHECK: bytecode.loc3 = loc("source.cc":10:8 to :12)
  // CHECK: bytecode.loc4 = loc("file.cc":0:0)
  // CHECK: bytecode.loc5 = loc("file.cc":42:0)
  bytecode.loc = loc("mysource.cc":10:8),
  bytecode.loc2 = loc("source.cc":10:8 to 12:4),
  bytecode.loc3 = loc("source.cc":10:8 to :12),
  bytecode.loc4 = loc("file.cc":0:0),
  bytecode.loc5 = loc("file.cc":42:0)
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

//===----------------------------------------------------------------------===//
// AffineMapAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestAffineMap
module @TestAffineMap attributes {
  // All binary expression types combined in a multi-result map.
  // CHECK: bytecode.allops_combined = affine_map<(d0, d1)[s0] -> ((d0 * 5 + d1) floordiv 4, (d0 + d1 * 3) ceildiv s0, (d0 + d1) mod s0)>

  // Binary operators (Add, Mul, Mod, FloorDiv, CeilDiv).
  // CHECK: bytecode.binop_add = affine_map<(d0, d1) -> (d0 + d1)>
  // CHECK: bytecode.binop_ceildiv = affine_map<(d0) -> (d0 ceildiv 8)>
  // CHECK: bytecode.binop_floordiv = affine_map<(d0) -> (d0 floordiv 4)>
  // CHECK: bytecode.binop_mod = affine_map<(d0) -> (d0 mod 3)>
  // CHECK: bytecode.binop_mul = affine_map<(d0) -> (d0 * 5)>

  // Dims, symbols, and constants.
  // CHECK: bytecode.dsc_const = affine_map<(d0) -> (d0 + 42)>
  // CHECK: bytecode.dsc_dim_sym = affine_map<(d0)[s0] -> (d0 + s0)>
  // CHECK: bytecode.dsc_multi_sym = affine_map<(d0, d1)[s0, s1] -> (d0 + s0, d1 + s1)>
  // CHECK: bytecode.dsc_sym_only = affine_map<()[s0] -> (s0 + 7)>

  // Empty map (zero dims, zero results).
  // CHECK: bytecode.empty_map = affine_map<() -> ()>

  // Identity maps.
  // CHECK: bytecode.identity_1d = affine_map<(d0) -> (d0)>
  // CHECK: bytecode.identity_3d = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  // CHECK: bytecode.identity_large = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15) -> (d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15)>

  // Multi-result maps.
  // CHECK: bytecode.multi_mix = affine_map<(d0, d1) -> (d0 + d1, d0 * 2, d1 mod 5)>
  // CHECK: bytecode.multi_mix_deep = affine_map<(d0, d1)[s0] -> (((d0 + d1) floordiv 4 + s0) ceildiv 2 + (d0 mod 3 + d1 * 5) floordiv 7)>
  // CHECK: bytecode.multi_sym = affine_map<(d0, d1)[s0] -> (d0 floordiv s0, d1 ceildiv s0, d0 + d1 + s0)>

  // Permutation maps.
  // CHECK: bytecode.perm_2d = affine_map<(d0, d1) -> (d1, d0)>
  // CHECK: bytecode.perm_3d = affine_map<(d0, d1, d2) -> (d2, d0, d1)>

  // Projected permutation maps.
  // CHECK: bytecode.projperm_4d = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
  // CHECK: bytecode.projperm_single = affine_map<(d0, d1, d2) -> (d2)>

  bytecode.allops_combined = affine_map<(d0, d1)[s0] -> ((d0 * 5 + d1) floordiv 4, (d0 + d1 * 3) ceildiv s0, (d0 + d1) mod s0)>,
  bytecode.binop_add = affine_map<(d0, d1) -> (d0 + d1)>,
  bytecode.binop_ceildiv = affine_map<(d0) -> (d0 ceildiv 8)>,
  bytecode.binop_floordiv = affine_map<(d0) -> (d0 floordiv 4)>,
  bytecode.binop_mod = affine_map<(d0) -> (d0 mod 3)>,
  bytecode.binop_mul = affine_map<(d0) -> (d0 * 5)>,
  bytecode.dsc_const = affine_map<(d0) -> (d0 + 42)>,
  bytecode.dsc_dim_sym = affine_map<(d0)[s0] -> (d0 + s0)>,
  bytecode.dsc_multi_sym = affine_map<(d0, d1)[s0, s1] -> (d0 + s0, d1 + s1)>,
  bytecode.dsc_sym_only = affine_map<()[s0] -> (s0 + 7)>,
  bytecode.empty_map = affine_map<() -> ()>,
  bytecode.identity_1d = affine_map<(d0) -> (d0)>,
  bytecode.identity_3d = affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
  bytecode.identity_large = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15) -> (d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15)>,
  bytecode.multi_mix = affine_map<(d0, d1) -> (d0 + d1, d0 * 2, d1 mod 5)>,
  bytecode.multi_mix_deep = affine_map<(d0, d1)[s0] -> (((d0 + d1) floordiv 4 + s0) ceildiv 2 + (d0 mod 3 + d1 * 5) floordiv 7)>,
  bytecode.multi_sym = affine_map<(d0, d1)[s0] -> (d0 floordiv s0, d1 ceildiv s0, d0 + d1 + s0)>,
  bytecode.perm_2d = affine_map<(d0, d1) -> (d1, d0)>,
  bytecode.perm_3d = affine_map<(d0, d1, d2) -> (d2, d0, d1)>,
  bytecode.projperm_4d = affine_map<(d0, d1, d2, d3) -> (d0, d2)>,
  bytecode.projperm_single = affine_map<(d0, d1, d2) -> (d2)>
} {}

//===----------------------------------------------------------------------===//
// IntegerSetAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestIntegerSetAttr
module @TestIntegerSetAttr attributes {
  // CHECK-DAG: bytecode.eq = affine_set<(d0) : (d0 == 0)>
  // CHECK-DAG: bytecode.ineq = affine_set<(d0) : (d0 >= 0)>
  // CHECK-DAG: bytecode.multi = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + s0 >= 0, d1 >= 0)>
  // CHECK-DAG: bytecode.eq_ineq = affine_set<(d0, d1) : (d0 == 0, d1 >= 0)>
  // CHECK-DAG: bytecode.neg_const = affine_set<(d0) : (d0 + 42 == 0)>
  // CHECK-DAG: bytecode.multi_sym = affine_set<(d0)[s0, s1] : (d0 + s0 >= 0, d0 + s1 >= 0)>
  // CHECK-DAG: bytecode.complex = affine_set<(d0, d1)[s0] : (d0 * 2 + d1 == 0, d0 - s0 >= 0, d1 mod 3 >= 0)>
  bytecode.eq = affine_set<(d0) : (d0 == 0)>,
  bytecode.ineq = affine_set<(d0) : (d0 >= 0)>,
  bytecode.multi = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + s0 >= 0, d1 >= 0)>,
  bytecode.eq_ineq = affine_set<(d0, d1) : (d0 == 0, d1 >= 0)>,
  bytecode.neg_const = affine_set<(d0) : (d0 + 42 == 0)>,
  bytecode.multi_sym = affine_set<(d0)[s0, s1] : (d0 + s0 >= 0, d0 + s1 >= 0)>,
  bytecode.complex = affine_set<(d0, d1)[s0] : (d0 * 2 + d1 == 0, d0 - s0 >= 0, d1 mod 3 >= 0)>
} {}

//===----------------------------------------------------------------------===//
// DictionaryAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestDictionaryAttr
module @TestDictionaryAttr attributes {
  // CHECK-DAG: bytecode.empty_dict = {}
  // CHECK-DAG: bytecode.nested_dict = {inner = {a = 1 : i32}}
  bytecode.empty_dict = {},
  bytecode.nested_dict = {inner = {a = 1 : i32}}
} {}

//===----------------------------------------------------------------------===//
// UnitAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestUnitAttr
module @TestUnitAttr attributes {
  // CHECK-DAG: bytecode.unit
  bytecode.unit
} {}

//===----------------------------------------------------------------------===//
// EmptyContainers
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestEmptyContainers
module @TestEmptyContainers attributes {
  // CHECK-DAG: bytecode.empty_arr = []
  // CHECK-DAG: bytecode.empty_dict = {}
  // CHECK-DAG: bytecode.empty_str = ""
  // CHECK-DAG: bytecode.empty_tuple = tuple<>
  // CHECK-DAG: bytecode.empty_func = () -> ()
  // CHECK-DAG: bytecode.empty_map = affine_map<() -> ()>
  // CHECK-DAG: bytecode.empty_dense_arr = array<i32>
  // CHECK-DAG: bytecode.empty_tensor = dense<> : tensor<0xi32>
  bytecode.empty_arr = [],
  bytecode.empty_dict = {},
  bytecode.empty_str = "",
  bytecode.empty_tuple = tuple<>,
  bytecode.empty_func = () -> (),
  bytecode.empty_map = affine_map<() -> ()>,
  bytecode.empty_dense_arr = array<i32>,
  bytecode.empty_tensor = dense<> : tensor<0xi32>
} {}

//===----------------------------------------------------------------------===//
// StridedLayoutAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @TestStridedLayoutAttr
module @TestStridedLayoutAttr attributes {
  // CHECK-DAG: bytecode.static_strides = strided<[3, 1]>
  // CHECK-DAG: bytecode.with_offset = strided<[3, 1], offset: 5>
  // CHECK-DAG: bytecode.dynamic = strided<[?, 1], offset: ?>
  bytecode.static_strides = strided<[3, 1]>,
  bytecode.with_offset = strided<[3, 1], offset: 5>,
  bytecode.dynamic = strided<[?, 1], offset: ?>
} {}

