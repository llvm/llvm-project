// REQUIRES: host-byteorder-little-endian
// RUN: mlir-opt -split-input-file -tosa-convert-integer-types-bitwidth='bitwidth-mappings=3:4,4:8,8:4,13:16,16:1,24:32,40:24,48:64,56:48,65:128,128:65' -mlir-print-op-generic %s | FileCheck %s
// RUN: mlir-opt -split-input-file -tosa-convert-integer-types-bitwidth='bitwidth-mappings=128:65,65:128,56:48,48:64,40:24,24:32,16:1,13:16,8:4,4:8,3:4' -mlir-print-op-generic %s | FileCheck %s

// Each mapping applies once to the original width, including cycles and
// destinations that are also sources. Reversing the list must have no effect.
// CHECK: "func.func"() <{function_type = (i4, i8, i4, i16, i1, i32, i24, i64, i48, i128, i65) -> (), sym_name = "types", sym_visibility = "private"}>
func.func private @types(i3, i4, i8, i13, i16, i24, i40, i48, i56, i65, i128)

// -----

// Narrow scalar, dense, sparse and array payloads, including i1.
// CHECK-LABEL: "builtin.module"() <{sym_name = "narrow_constants"}>
// CHECK: test.a_scalar = -7 : i4
// CHECK-SAME: test.b_all_ones = -1 : i4
// CHECK-SAME: test.c_array = array<i24: -8388608, 8388607, -1, 0>
// CHECK-SAME: test.d_splat = dense<-7> : tensor<3xi4>
// CHECK-SAME: test.e_high_bits = dense<[-8, 7, -1, 0]> : tensor<4xi4>
// CHECK-SAME: test.f_sparse = sparse<{{\[\[}}1], [3]], [-1, 0]> : tensor<4xi48>
// CHECK-SAME: test.g_i1_array = array<i1: true, false, true>
// CHECK-SAME: test.h_i1_dense = dense<[true, false, true]> : tensor<3xi1>
// CHECK-SAME: test.i_types = [i24, i48, i65]
// CHECK-SAME: test.j_wide_sparse = sparse<{{\[\[}}1], [3]], [-1, 18446744073709551615]> : tensor<4xi65>
module @narrow_constants attributes {
  test.a_scalar = 25 : i8,
  test.b_all_ones = 255 : i8,
  test.c_array = array<i40: 8388608, 25165823, -1, -16777216>,
  test.d_splat = dense<25> : tensor<3xi8>,
  test.e_high_bits = dense<[8, 23, 255, 16]> : tensor<4xi8>,
  test.f_sparse = sparse<[[1], [3]], [-1, 281474976710656]> : tensor<4xi56>,
  test.g_i1_array = array<i16: -1, 2, 3>,
  test.h_i1_dense = dense<[-1, 2, 3]> : tensor<3xi16>,
  test.i_types = [i40, i56, i128],
  test.j_wide_sparse = sparse<[[1], [3]], [-1, 18446744073709551615]> : tensor<4xi128>
} {}

// -----

// Widen sub-byte, non-power-of-two and greater-than-64-bit payloads.
// CHECK-LABEL: "builtin.module"() <{sym_name = "widen_constants"}>
// CHECK: test.array = array<i32: -8388608, 8388607>
// CHECK-SAME: test.array_ones = array<i32: -1>
// CHECK-SAME: test.dense = dense<[-4, -1, 0, 3]> : tensor<4xi4>
// CHECK-SAME: test.mid_bounds = dense<[-4096, 4095]> : tensor<2xi16>
// CHECK-SAME: test.mid_ones = -1 : i16
// CHECK-SAME: test.scalar = -4 : i4
// CHECK-SAME: test.sparse = sparse<{{\[\[}}1], [3]], [-2, 1]> : tensor<4xi4>
// CHECK-SAME: test.top_bits = -1 : i4
// CHECK-SAME: test.wide = dense<[-1, 18446744073709551615]> : tensor<2xi128>
// CHECK-SAME: test.wide_ones = -1 : i128
module @widen_constants attributes {
  test.array = array<i24: -8388608, 8388607>,
  test.array_ones = array<i24: 16777215>,
  test.dense = dense<[-4, -1, 0, 3]> : tensor<4xi3>,
  test.mid_bounds = dense<[-4096, 4095]> : tensor<2xi13>,
  test.mid_ones = 8191 : i13,
  test.scalar = -4 : i3,
  test.sparse = sparse<[[1], [3]], [-2, 1]> : tensor<4xi3>,
  test.top_bits = 7 : i3,
  test.wide = dense<[-1, 18446744073709551615]> : tensor<2xi65>,
  test.wide_ones = 36893488147419103231 : i65
} {}

// -----

// Map every nested type once, including when resource data is unavailable.
// CHECK-LABEL: "builtin.module"() <{sym_name = "mapping_encodings"}>
// CHECK: test.type = tensor<2xi4, {test.widths = [i4, i8, i4]}>
// CHECK-SAME: test.unresolved_resource = dense_resource<missing> : tensor<2xi7, {test.widths = [i4, i8, i4]}>
module @mapping_encodings attributes {
  test.type = tensor<2xi3, {test.widths = [i3, i4, i8]}>,
  test.unresolved_resource = dense_resource<missing> : tensor<2xi7, {test.widths = [i3, i4, i8]}>
} {}

// -----

// CHECK-LABEL: "builtin.module"() <{sym_name = "narrow_resources"}>
// CHECK: test.a_i4 = dense_resource<[[I4:[a-zA-Z0-9_]+]]> : tensor<4xi4>
// CHECK-SAME: test.b_i24 = dense_resource<[[I24:[a-zA-Z0-9_]+]]> : tensor<2xi24>
// CHECK-SAME: test.c_i48 = dense_resource<[[I48:[a-zA-Z0-9_]+]]> : tensor<2xi48>
// CHECK-SAME: test.d_i65 = dense_resource<[[I65:[a-zA-Z0-9_]+]]> : tensor<2xi65>
// CHECK-SAME: test.e_i1 = dense_resource<[[I1:[a-zA-Z0-9_]+]]> : tensor<3xi1>
module @narrow_resources attributes {
  test.a_i4 = dense_resource<to4> : tensor<4xi8>,
  test.b_i24 = dense_resource<to24> : tensor<2xi40>,
  test.c_i48 = dense_resource<to48> : tensor<2xi56>,
  test.d_i65 = dense_resource<to65> : tensor<2xi128>,
  test.e_i1 = dense_resource<to1> : tensor<3xi16>
} {}

{-# dialect_resources: { builtin: {
  to4: "0x010000000917FFF0",
  to24: "0x01000000FFFFFFFFFFFFFFFFFF00",
  to48: "0x01000000FFFFFFFFFFFFFF00000000000001",
  to65: "0x01000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000",
  to1: "0x01000000FFFF02000300"
  // CHECK-DAG: [[I4]]: "0x0100000009070F00"
  // CHECK-DAG: [[I24]]: "0x04000000FFFFFFFFFFFF"
  // CHECK-DAG: [[I48]]: "0x08000000FFFFFFFFFFFF000000000000"
  // CHECK-DAG: [[I65]]: "0x10000000FFFFFFFFFFFFFFFF01FFFFFFFFFFFFFFFF00"
  // CHECK-DAG: [[I1]]: "0x01000000010001"
} } #-}

// -----

// Widen resource elements and their encodings using the original types.
// CHECK-LABEL: "builtin.module"() <{sym_name = "widen_resources"}>
// CHECK: test.a_subbyte = dense_resource<[[SUBBYTE:[a-zA-Z0-9_]+]]> : tensor<4xi4, {test.widths = [i4, i8, i4]}>
// CHECK-SAME: test.b_i16 = dense_resource<[[I16:[a-zA-Z0-9_]+]]> : tensor<2xi16>
// CHECK-SAME: test.c_i32 = dense_resource<[[I32:[a-zA-Z0-9_]+]]> : tensor<2xi32>
// CHECK-SAME: test.d_i128 = dense_resource<[[I128:[a-zA-Z0-9_]+]]> : tensor<2xi128>
module @widen_resources attributes {
  test.a_subbyte = dense_resource<subbyte> : tensor<4xi3, {test.widths = [i3, i4, i8]}>,
  test.b_i16 = dense_resource<to16> : tensor<2xi13>,
  test.c_i32 = dense_resource<to32> : tensor<2xi24>,
  test.d_i128 = dense_resource<to128> : tensor<2xi65>
} {}

{-# dialect_resources: { builtin: {
  // [-4, -1, 0, 3], with one byte per i3 value.
  subbyte: "0x0100000004070003",
  // [-4096, 4095], with two bytes per i13 value.
  to16: "0x010000000010FF0F",
  // [-8388608, 8388607], with three bytes per i24 value.
  to32: "0x01000000000080FFFF7F",
  // [-1, 2^64 - 1], with nine bytes per i65 value.
  to128: "0x01000000FFFFFFFFFFFFFFFF01FFFFFFFFFFFFFFFF00"
  // CHECK-DAG: [[SUBBYTE]]: "0x010000000C0F0003"
  // CHECK-DAG: [[I16]]: "0x0200000000F0FF0F"
  // CHECK-DAG: [[I32]]: "0x04000000000080FFFFFF7F00"
  // CHECK-DAG: [[I128]]: "0x10000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000"
} } #-}
