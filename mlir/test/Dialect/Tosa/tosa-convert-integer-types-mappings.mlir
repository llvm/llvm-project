// REQUIRES: host-byteorder-little-endian
// RUN: mlir-opt -split-input-file -tosa-convert-integer-types='bitwidth-mappings=1:8,3:4,4:8,8:4,13:16,16:1,24:32,32:24,48:64,64:48,65:128,128:65' -mlir-print-op-generic %s | FileCheck %s --check-prefixes=CHECK,SCALAR '-D$WIDTH=24'
// RUN: mlir-opt -split-input-file -tosa-convert-integer-types='bitwidth-mappings=128:65,65:128,64:48,48:64,32:24,24:32,16:1,13:16,8:4,4:8,3:4,1:8' -mlir-print-op-generic %s | FileCheck %s --check-prefixes=CHECK,SCALAR '-D$WIDTH=24'
// RUN: mlir-opt -split-input-file -tosa-convert-integer-types='bitwidth-mappings=1:8,32:64' -mlir-print-op-generic %s | FileCheck %s --check-prefix=SCALAR '-D$WIDTH=64'

// Each mapping applies once to the original width, including cycles and
// destinations that are also sources. Reversing the list must have no effect.
// CHECK: "func.func"() <{function_type = (i8, i4, si4, ui4, i8, i4, i16, i1, i32, i24, i64, i48, i128, i65) -> (), sym_name = "types", sym_visibility = "private"}>
func.func private @types(i1, i3, si3, ui3, i4, i8, i13, i16, i24, i32, i48, i64, i65, i128)

// -----

// Conversion truncates, including when values do not fit in the destination.
// CHECK: "func.func"() <{function_type = (tensor<4xi4>, tensor<2xi48>) -> tensor<4xi4>, sym_name = "callee", sym_visibility = "private"}>
func.func private @callee(tensor<4xi8>, tensor<2xi64>) -> tensor<4xi8>
// CHECK-LABEL: "func.func"() <{function_type = (tensor<2xi48>) -> tensor<4xi4>, sym_name = "caller"}>
func.func @caller(%arg: tensor<2xi64>) -> tensor<4xi8> {
  // CHECK: "tosa.const"() <{values = dense<[-7, 7, -1, 0]> : tensor<4xi4>}>
  %0 = tosa.const values(dense<[9, 23, -1, -16]> : tensor<4xi8>) : () -> tensor<4xi8>
  // CHECK: "func.call"(%{{.*}}, %{{.*}}) <{callee = @callee}> : (tensor<4xi4>, tensor<2xi48>) -> tensor<4xi4>
  %1 = func.call @callee(%0, %arg) : (tensor<4xi8>, tensor<2xi64>) -> tensor<4xi8>
  return %1 : tensor<4xi8>
}

// -----

// Narrow scalar, dense, sparse and array payloads, including i1.
// CHECK-LABEL: "builtin.module"() <{sym_name = "narrow_constants"}>
// CHECK: test.a_scalar = -7 : si4
// CHECK-SAME: test.b_unsigned = 15 : ui4
// CHECK-SAME: test.c_array = array<i24: -8388608, 8388607, -1, 0>
// CHECK-SAME: test.d_splat = dense<-7> : tensor<3xsi4>
// CHECK-SAME: test.e_unsigned = dense<[8, 7, 15, 0]> : tensor<4xui4>
// CHECK-SAME: test.f_sparse = sparse<{{\[\[}}1], [3]], [-1, 0]> : tensor<4xi48>
// CHECK-SAME: test.g_i1_array = array<i1: true, false, true>
// CHECK-SAME: test.h_i1_dense = dense<[true, false, true]> : tensor<3xi1>
// CHECK-SAME: test.i_types = [i24, si48, ui65]
// CHECK-SAME: test.j_wide_sparse = sparse<{{\[\[}}1], [3]], [-1, 18446744073709551615]> : tensor<4xi65>
module @narrow_constants attributes {
  test.a_scalar = 25 : si8,
  test.b_unsigned = 255 : ui8,
  test.c_array = array<i32: 8388608, 25165823, -1, -16777216>,
  test.d_splat = dense<25> : tensor<3xsi8>,
  test.e_unsigned = dense<[8, 23, 255, 16]> : tensor<4xui8>,
  test.f_sparse = sparse<[[1], [3]], [-1, 281474976710656]> : tensor<4xi64>,
  test.g_i1_array = array<i16: -1, 2, 3>,
  test.h_i1_dense = dense<[-1, 2, 3]> : tensor<3xi16>,
  test.i_types = [i32, si64, ui128],
  test.j_wide_sparse = sparse<[[1], [3]], [-1, 18446744073709551615]> : tensor<4xi128>
} {}

// -----

// The 64:48 mapping must truncate stored values, but preserve coordinates.
// Bit 48 distinguishes this index from zero even without printed index types.
// CHECK-LABEL: "builtin.module"() <{sym_name = "sparse_indices"}>
// CHECK: test.sparse = sparse<{{\[\[}}0], [281474976710656]], [-1, 0]> : tensor<281474976710657xi48>
module @sparse_indices attributes {
  test.sparse = sparse<[[0], [281474976710656]], [-1, 281474976710656]> : tensor<281474976710657xi64>
} {}

// -----

// Widen sub-byte, non-power-of-two and greater-than-64-bit payloads.
// CHECK-LABEL: "builtin.module"() <{sym_name = "widen_constants"}>
// CHECK: test.array = array<i32: -8388608, 8388607>
// CHECK-SAME: test.array_unsigned = array<ui32: 16777215>
// CHECK-SAME: test.dense = dense<[-4, -1, 0, 3]> : tensor<4xi4>
// CHECK-SAME: test.mid_signed = dense<[-4096, 4095]> : tensor<2xi16>
// CHECK-SAME: test.mid_unsigned = 8191 : ui16
// CHECK-SAME: test.scalar = -4 : si4
// CHECK-SAME: test.sparse = sparse<{{\[\[}}1], [3]], [-2, 1]> : tensor<4xi4>
// CHECK-SAME: test.unsigned = 7 : ui4
// CHECK-SAME: test.wide = dense<[-1, 18446744073709551615]> : tensor<2xi128>
// CHECK-SAME: test.wide_unsigned = 36893488147419103231 : ui128
module @widen_constants attributes {
  test.array = array<i24: -8388608, 8388607>,
  test.array_unsigned = array<ui24: 16777215>,
  test.dense = dense<[-4, -1, 0, 3]> : tensor<4xi3>,
  test.mid_signed = dense<[-4096, 4095]> : tensor<2xi13>,
  test.mid_unsigned = 8191 : ui13,
  test.scalar = -4 : si3,
  test.sparse = sparse<[[1], [3]], [-2, 1]> : tensor<4xi3>,
  test.unsigned = 7 : ui3,
  test.wide = dense<[-1, 18446744073709551615]> : tensor<2xi65>,
  test.wide_unsigned = 36893488147419103231 : ui65
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
  test.b_i24 = dense_resource<to24> : tensor<2xi32>,
  test.c_i48 = dense_resource<to48> : tensor<2xi64>,
  test.d_i65 = dense_resource<to65> : tensor<2xi128>,
  test.e_i1 = dense_resource<to1> : tensor<3xi16>
} {}

{-# dialect_resources: { builtin: {
  to4: "0x010000000917FFF0",
  to24: "0x01000000FFFFFFFFFFFFFF00",
  to48: "0x01000000FFFFFFFFFFFFFFFF0000000000000100",
  to65: "0x01000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000",
  to1: "0x01000000FFFF02000300"
  // CHECK-DAG: [[I4]]: "0x0100000009070F00"
  // CHECK-DAG: [[I24]]: "0x04000000FFFFFFFFFFFF"
  // CHECK-DAG: [[I48]]: "0x08000000FFFFFFFFFFFF000000000000"
  // CHECK-DAG: [[I65]]: "0x08000000FFFFFFFFFFFFFFFF01FFFFFFFFFFFFFFFF00"
  // CHECK-DAG: [[I1]]: "0x01000000010001"
} } #-}

// -----

// Widen resource elements and their encodings using the original types.
// CHECK-LABEL: "builtin.module"() <{sym_name = "widen_resources"}>
// CHECK: test.a_subbyte = dense_resource<[[SUBBYTE:[a-zA-Z0-9_]+]]> : tensor<4xi4, {test.widths = [i4, i8, i4]}>
// CHECK-SAME: test.b_i16 = dense_resource<[[I16:[a-zA-Z0-9_]+]]> : tensor<2xi16>
// CHECK-SAME: test.c_i32 = dense_resource<[[I32:[a-zA-Z0-9_]+]]> : tensor<2xi32>
// CHECK-SAME: test.d_i128 = dense_resource<[[I128:[a-zA-Z0-9_]+]]> : tensor<2xi128>
// CHECK-SAME: test.e_unsigned = dense_resource<[[UNSIGNED:[a-zA-Z0-9_]+]]> : tensor<3xui4>
module @widen_resources attributes {
  test.a_subbyte = dense_resource<subbyte> : tensor<4xi3, {test.widths = [i3, i4, i8]}>,
  test.b_i16 = dense_resource<to16> : tensor<2xi13>,
  test.c_i32 = dense_resource<to32> : tensor<2xi24>,
  test.d_i128 = dense_resource<to128> : tensor<2xi65>,
  test.e_unsigned = dense_resource<unsigned_data> : tensor<3xui3>
} {}

{-# dialect_resources: { builtin: {
  // [-4, -1, 0, 3], with one byte per i3 value.
  subbyte: "0x0100000004070003",
  // [-4096, 4095], with two bytes per i13 value.
  to16: "0x010000000010FF0F",
  // [-8388608, 8388607], with three bytes per i24 value.
  to32: "0x01000000000080FFFF7F",
  // [-1, 2^64 - 1], with nine bytes per i65 value.
  to128: "0x01000000FFFFFFFFFFFFFFFF01FFFFFFFFFFFFFFFF00",
  // [0, 4, 7].
  unsigned_data: "0x01000000000407"
  // CHECK-DAG: [[SUBBYTE]]: "0x010000000C0F0003"
  // CHECK-DAG: [[I16]]: "0x0200000000F0FF0F"
  // CHECK-DAG: [[I32]]: "0x04000000000080FFFFFF7F00"
  // CHECK-DAG: [[I128]]: "0x08000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000"
  // CHECK-DAG: [[UNSIGNED]]: "0x01000000000407"
} } #-}

// -----

// Structural attributes have operation-defined storage types. Mapping i64 to
// i48 must preserve the required DenseI64ArrayAttr properties, even though
// standalone dense arrays can be converted. All tensor elements here are f32
// to isolate changes to structural attributes from tensor type conversion.
// Generic printing exposes the attribute storage types.
// The discardable test.pad attribute shares the original padding array but
// must still be converted to i48.

// CHECK: "tosa.conv2d"
// CHECK-SAME: acc_type = f32
// CHECK-SAME: dilation = array<i64: 1, 1>
// CHECK-SAME: pad = array<i64: 0, 0, 0, 0>
// CHECK-SAME: stride = array<i64: 1, 1>
// CHECK-SAME: test.pad = array<i48: 0, 0, 0, 0>
// CHECK-SAME: -> tensor<1x3x3x1xf32>
func.func @conv2d(%input: tensor<1x3x3x1xf32>, %weight: tensor<1x1x1x1xf32>, %bias: tensor<1xf32>, %izp: tensor<1xf32>, %wzp: tensor<1xf32>) -> tensor<1x3x3x1xf32> {
  %0 = tosa.conv2d %input, %weight, %bias, %izp, %wzp pad([0, 0, 0, 0]) stride([1, 1]) dilation([1, 1]) acc_type(f32) {test.pad = array<i64: 0, 0, 0, 0>} : (tensor<1x3x3x1xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x3x3x1xf32>
  return %0 : tensor<1x3x3x1xf32>
}

// -----

// CHECK: "tosa.avg_pool2d"
// CHECK-SAME: acc_type = f32
// CHECK-SAME: kernel = array<i64: 2, 2>
// CHECK-SAME: pad = array<i64: 0, 1, 0, 1>
// CHECK-SAME: stride = array<i64: 1, 1>
// CHECK-SAME: -> tensor<1x7x7x9xf32>
func.func @avg_pool2d(%input: tensor<1x7x7x9xf32>, %izp: tensor<1xf32>, %ozp: tensor<1xf32>) -> tensor<1x7x7x9xf32> {
  %0 = tosa.avg_pool2d %input, %izp, %ozp kernel([2, 2]) stride([1, 1]) pad([0, 1, 0, 1]) acc_type(f32) : (tensor<1x7x7x9xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x7x7x9xf32>
  return %0 : tensor<1x7x7x9xf32>
}

// -----

// TOSA flags keep their boolean storage, while the same values in metadata
// and arithmetic constants follow the mapping. Check true and false explicitly
// because dropping an optional flag can silently restore its default value.
// SCALAR-LABEL: sym_name = "boolean_flags"
func.func @boolean_flags(%arg: tensor<2xi32>) -> tensor<2xi32> {
  // SCALAR: "tosa.cast"
  // SCALAR-SAME: input_unsigned = true
  // SCALAR-SAME: test.flag = -1 : i8
  // SCALAR-SAME: test.nested = {flag = -1 : i8}
  %0 = tosa.cast %arg input_unsigned(true) {test.flag = true, test.nested = {flag = true}} : (tensor<2xi32>) -> tensor<2xi32>
  // SCALAR: "tosa.cast"
  // SCALAR-SAME: input_unsigned = false
  // SCALAR-SAME: test.flag = 0 : i8
  %1 = tosa.cast %0 input_unsigned(false) {test.flag = false} : (tensor<2xi32>) -> tensor<2xi32>
  // SCALAR: "arith.constant"() <{value = -1 : i8}>
  %true = arith.constant true
  // SCALAR: "arith.constant"() <{value = 0 : i8}>
  %false = arith.constant false
  return %1 : tensor<2xi32>
}

// -----

// Axes retain their required i32 storage during both widening and narrowing.
// Scalar clamp bounds, discardable attributes and nested metadata still change.
// SCALAR-LABEL: sym_name = "scalar_attributes"
func.func @scalar_attributes(%arg: tensor<2x3xi32>) -> tensor<2x1xi32> {
  // SCALAR: "tosa.clamp"
  // SCALAR-SAME: max_val = 7 : i[[$WIDTH]], min_val = 0 : i[[$WIDTH]]
  // SCALAR-SAME: -> tensor<2x3xi[[$WIDTH]]>
  %0 = tosa.clamp %arg min_val(0 : i32) max_val(7 : i32) : (tensor<2x3xi32>) -> tensor<2x3xi32>
  // SCALAR: "tosa.reduce_sum"
  // SCALAR-SAME: axis = 1 : i32
  // SCALAR-SAME: test.axis = 1 : i[[$WIDTH]]
  // SCALAR-SAME: test.nested = {axis = 1 : i[[$WIDTH]]}
  // SCALAR-SAME: -> tensor<2x1xi[[$WIDTH]]>
  %1 = tosa.reduce_sum %0 axis(1) {test.axis = 1 : i32, test.nested = {axis = 1 : i32}} : (tensor<2x3xi32>) -> tensor<2x1xi32>
  return %1 : tensor<2x1xi32>
}
