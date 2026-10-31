// REQUIRES: host-byteorder-little-endian
// RUN: mlir-opt -split-input-file -pass-pipeline='builtin.module(tosa-convert-integer-types-bitwidth{bitwidth-mappings=4:8,48:64})' %s | FileCheck %s --implicit-check-not=i4 --implicit-check-not=unrealized_conversion_cast --implicit-check-not=tosa.cast
// RUN: mlir-opt -split-input-file -tosa-convert-integer-types-bitwidth='bitwidth-mappings=4:8,48:64' %s | mlir-opt -split-input-file -tosa-convert-integer-types-bitwidth='bitwidth-mappings=4:8,48:64' | FileCheck %s --implicit-check-not=i4 --implicit-check-not=unrealized_conversion_cast --implicit-check-not=tosa.cast

// CHECK-LABEL: func.func @clamp
func.func @clamp(%arg: tensor<4xi48>) -> tensor<4xi48> {
  // CHECK: tosa.clamp %{{.*}} min_val(-140737488355328 : i64) max_val(140737488355327 : i64) {test.unrelated_attribute_that_must_be_preserved = "keep"} : (tensor<4xi64>) -> tensor<4xi64>
  %0 = tosa.clamp %arg min_val(-140737488355328 : i48) max_val(140737488355327 : i48) {test.unrelated_attribute_that_must_be_preserved = "keep"} : (tensor<4xi48>) -> tensor<4xi48>
  return %0 : tensor<4xi48>
}

// -----

// CHECK-LABEL: func.func @cond_if
func.func @cond_if(%cond: tensor<i1>, %arg: tensor<2xi48>) -> tensor<2xi48> {
  // CHECK: %[[IF:.*]] = tosa.cond_if %arg0
  // CHECK-SAME: (tensor<2xi64>) -> tensor<2xi64>
  %0 = tosa.cond_if %cond (%x = %arg) : tensor<i1> (tensor<2xi48>) -> tensor<2xi48> {
  // CHECK: ^bb0(%[[THEN:.*]]: tensor<2xi64>):
  ^bb0(%x: tensor<2xi48>):
    // CHECK: %[[ADD:.*]] = tosa.add %[[THEN]], %[[THEN]] : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
    %1 = tosa.add %x, %x : (tensor<2xi48>, tensor<2xi48>) -> tensor<2xi48>
    // CHECK: tosa.yield %[[ADD]] : tensor<2xi64>
    tosa.yield %1 : tensor<2xi48>
  } else {
  // CHECK: ^bb0(%[[ELSE:.*]]: tensor<2xi64>):
  ^bb0(%x: tensor<2xi48>):
    // CHECK: tosa.yield %[[ELSE]] : tensor<2xi64>
    tosa.yield %x : tensor<2xi48>
  }
  // CHECK: return %[[IF]] : tensor<2xi64>
  return %0 : tensor<2xi48>
}

// -----

// CHECK-LABEL: func.func @conv_rescale(
// CHECK-SAME: %{{.*}}: tensor<1x3x3x1xi16>, %{{.*}}: tensor<1x1x1x1xi8>) -> tensor<1x3x3x1xi8>
func.func @conv_rescale(%input: tensor<1x3x3x1xi16>, %weight: tensor<1x1x1x1xi8>) -> tensor<1x3x3x1xi8> {
  // CHECK: %[[BIAS:.*]] = tosa.const values(dense<0> : tensor<1xi64>) : () -> tensor<1xi64>
  %bias = tosa.const values(dense<0> : tensor<1xi48>) : () -> tensor<1xi48>
  %izp = tosa.const values(dense<0> : tensor<1xi16>) : () -> tensor<1xi16>
  %wzp = tosa.const values(dense<0> : tensor<1xi8>) : () -> tensor<1xi8>
  // CHECK: %[[CONV:.*]] = tosa.conv2d %{{.*}}, %{{.*}}, %[[BIAS]], %{{.*}}, %{{.*}} pad([0, 0, 0, 0]) stride([1, 1]) dilation([1, 1]) acc_type(i64) : (tensor<1x3x3x1xi16>, tensor<1x1x1x1xi8>, tensor<1xi64>, tensor<1xi16>, tensor<1xi8>) -> tensor<1x3x3x1xi64>
  %0 = tosa.conv2d %input, %weight, %bias, %izp, %wzp pad([0, 0, 0, 0]) stride([1, 1]) dilation([1, 1]) acc_type(i48) : (tensor<1x3x3x1xi16>, tensor<1x1x1x1xi8>, tensor<1xi48>, tensor<1xi16>, tensor<1xi8>) -> tensor<1x3x3x1xi48>
  %mult = tosa.const values(dense<16384> : tensor<1xi16>) : () -> tensor<1xi16>
  %shift = tosa.const values(dense<15> : tensor<1xi8>) : () -> tensor<1xi8>
  // CHECK: %[[ZP:.*]] = tosa.const values(dense<0> : tensor<1xi64>) : () -> tensor<1xi64>
  %rzp = tosa.const values(dense<0> : tensor<1xi48>) : () -> tensor<1xi48>
  // CHECK: tosa.rescale %[[CONV]], %{{.*}}, %{{.*}}, %[[ZP]], %{{.*}} scale32(false) rounding_mode<SINGLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<1x3x3x1xi64>, tensor<1xi16>, tensor<1xi8>, tensor<1xi64>, tensor<1xi8>) -> tensor<1x3x3x1xi8>
  %1 = tosa.rescale %0, %mult, %shift, %rzp, %wzp scale32(false) rounding_mode<SINGLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<1x3x3x1xi48>, tensor<1xi16>, tensor<1xi8>, tensor<1xi48>, tensor<1xi8>) -> tensor<1x3x3x1xi8>
  return %1 : tensor<1x3x3x1xi8>
}

// -----

// Recurse into nested attributes even when operand and result types are unchanged.
// CHECK-LABEL: func.func @attributes_only
func.func @attributes_only(%arg: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK: tosa.identity %arg0 {test.nested = [{min = -140737488355328 : i64, top_bits = -1 : i64, type = tensor<2xi64>}], test.unchanged = 48 : i32} : (tensor<2xi32>) -> tensor<2xi32>
  %0 = tosa.identity %arg {test.nested = [{min = -140737488355328 : i48, top_bits = -1 : i48, type = tensor<2xi48>}], test.unchanged = 48 : i32} : (tensor<2xi32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// -----

// Mixed types and multiple results; preserve function and call attributes.
// CHECK-LABEL: func.func private @mixed(
// CHECK-SAME: %[[A:.*]]: tensor<?xi64>, %[[B:.*]]: tensor<*xi64>, %[[C:.*]]: tensor<2xi32>) -> (tensor<*xi64>, tensor<?xi64>, tensor<2xi32>)
func.func private @mixed(%a: tensor<?xi48>, %b: tensor<*xi48>, %c: tensor<2xi32>) -> (tensor<*xi48>, tensor<?xi48>, tensor<2xi32>) {
  // CHECK: return %[[B]], %[[A]], %[[C]] : tensor<*xi64>, tensor<?xi64>, tensor<2xi32>
  return %b, %a, %c : tensor<*xi48>, tensor<?xi48>, tensor<2xi32>
}
// CHECK-LABEL: func.func @mixed_caller(
// CHECK-SAME: %[[A:.*]]: tensor<?xi64>, %[[B:.*]]: tensor<*xi64>, %[[C:.*]]: tensor<2xi32>) -> (tensor<*xi64>, tensor<?xi64>, tensor<2xi32>) attributes {test.unrelated_attribute_that_must_be_preserved = "function"}
func.func @mixed_caller(%a: tensor<?xi48>, %b: tensor<*xi48>, %c: tensor<2xi32>) -> (tensor<*xi48>, tensor<?xi48>, tensor<2xi32>) attributes {test.unrelated_attribute_that_must_be_preserved = "function"} {
  // CHECK: %[[CALL:.*]]:3 = call @mixed(%[[A]], %[[B]], %[[C]]) {test.unrelated_attribute_that_must_be_preserved = "call"} : (tensor<?xi64>, tensor<*xi64>, tensor<2xi32>) -> (tensor<*xi64>, tensor<?xi64>, tensor<2xi32>)
  %0:3 = func.call @mixed(%a, %b, %c) {test.unrelated_attribute_that_must_be_preserved = "call"} : (tensor<?xi48>, tensor<*xi48>, tensor<2xi32>) -> (tensor<*xi48>, tensor<?xi48>, tensor<2xi32>)
  // CHECK: return %[[CALL]]#0, %[[CALL]]#1, %[[CALL]]#2 : tensor<*xi64>, tensor<?xi64>, tensor<2xi32>
  return %0#0, %0#1, %0#2 : tensor<*xi48>, tensor<?xi48>, tensor<2xi32>
}

// -----

// Convert function, argument and result metadata along with the signature.
// CHECK-LABEL: func.func @function_attributes(
// CHECK-SAME: %[[ARG:.*]]: tensor<2xi64> {test.type = tensor<2xi64>}) -> (tensor<2xi64> {test.type = tensor<2xi64>}) attributes {test.value = -1 : i64}
func.func @function_attributes(%arg: tensor<2xi48> {test.type = tensor<2xi48>}) -> (tensor<2xi48> {test.type = tensor<2xi48>}) attributes {test.value = -1 : i48} {
  // CHECK: return %[[ARG]] : tensor<2xi64>
  return %arg : tensor<2xi48>
}

// -----

// Preserve a memref's non-default strides and offset while widening its elements.
// CHECK-LABEL: func.func @strided_memref(
// CHECK-SAME: %[[ARG:.*]]: memref<2x3xi64, strided<[5, 1], offset: 2>>) -> memref<2x3xi64, strided<[5, 1], offset: 2>>
func.func @strided_memref(%arg: memref<2x3xi48, strided<[5, 1], offset: 2>>) -> memref<2x3xi48, strided<[5, 1], offset: 2>> {
  // CHECK: return %[[ARG]] : memref<2x3xi64, strided<[5, 1], offset: 2>>
  return %arg : memref<2x3xi48, strided<[5, 1], offset: 2>>
}

// -----

// Widen an arithmetic producer and its TOSA user inside a non-TOSA container.
// CHECK-LABEL: func.func @nested_tosa()
func.func @nested_tosa() {
  // CHECK: scf.execute_region {
  scf.execute_region {
    // CHECK: %[[C:.*]] = arith.constant dense<-1> : tensor<2xi64>
    %c = arith.constant dense<-1> : tensor<2xi48>
    // CHECK: tosa.identity %[[C]] : (tensor<2xi64>) -> tensor<2xi64>
    %0 = tosa.identity %c : (tensor<2xi48>) -> tensor<2xi48>
    // CHECK-NEXT: scf.yield
    scf.yield
  }
  // CHECK: return
  return
}

// -----

// Only the selected i4 and i48 widths change. Arbitrary widths such as i3
// remain unchanged.
// CHECK: func.func private @types(i1, i3, i8, i7, i8, i9, i16, i24, i32, i64, i64, i128) -> (i3, i7)
func.func private @types(i1, i3, i4, i7, i8, i9, i16, i24, i32, i48, i64, i128) -> (i3, i7)

// -----

// Constant payloads and resource storage.

// CHECK-LABEL: func.func @subbyte_constants()
// CHECK-SAME: -> (tensor<6xi8>, tensor<2x3xi8>)
func.func @subbyte_constants() -> (tensor<6xi4>, tensor<2x3xi4>) {
  // CHECK: tosa.const values(dense<[-8, -7, -1, 0, 1, 7]> : tensor<6xi8>)
  %a = tosa.const values(dense<[-8, -7, -1, 0, 1, 7]> : tensor<6xi4>) : () -> tensor<6xi4>
  // CHECK: tosa.const values(dense<-7> : tensor<2x3xi8>)
  %b = tosa.const values(dense<-7> : tensor<2x3xi4>) : () -> tensor<2x3xi4>
  return %a, %b : tensor<6xi4>, tensor<2x3xi4>
}

// -----

// Widen scalar and dense attributes, including nested scalar values.
// CHECK-LABEL: func.func @attributes(
// CHECK-SAME: tensor<2xi8, {type = i8}>) -> tensor<2xi8, {type = i8}>
func.func @attributes(%arg: tensor<2xi4, {type = i4}>) -> tensor<2xi4, {type = i4}> {
  // CHECK: tosa.identity %arg0
  // CHECK-SAME: test.nested = [{min = -7 : i8, negative = -1 : i8}]
  // CHECK-SAME: test.values = dense<[-7, -1, 7]> : tensor<3xi8>
  %0 = tosa.identity %arg {
    test.nested = [{min = -7 : i4, negative = -1 : i4}],
    test.values = dense<[-7, -1, 7]> : tensor<3xi4>
  } : (tensor<2xi4, {type = i4}>) -> tensor<2xi4, {type = i4}>
  return %0 : tensor<2xi4, {type = i4}>
}

// -----

// i4 resources use one byte per element. Preserve the original blob for an i8
// view, share the widened resource, and retain storage for repeated values.
// CHECK: module attributes {test.bytes = dense_resource<values> : tensor<6xi8>, test.original = dense_resource<[[VALUES:[a-zA-Z0-9_]+]]> : tensor<6xi8>}
module attributes {
  test.bytes = dense_resource<values> : tensor<6xi8>,
  test.original = dense_resource<values> : tensor<6xi4>
} {
  // CHECK: func.func @resources()
  // CHECK-SAME: attributes {test.repeated = dense_resource<[[REPEATED:[a-zA-Z0-9_]+]]> : tensor<3xi8>}
  func.func @resources() attributes {test.repeated = dense_resource<repeated> : tensor<3xi4>} {
    // CHECK: tosa.const values(dense_resource<[[VALUES]]> : tensor<6xi8>)
    %a = tosa.const values(dense_resource<values> : tensor<6xi4>) : () -> tensor<6xi4>
    return
  }
  // CHECK: func.func @shared_resource()
  func.func @shared_resource() {
    // CHECK: tosa.const values(dense_resource<[[VALUES]]> : tensor<6xi8>)
    %0 = tosa.const values(dense_resource<values> : tensor<6xi4>) : () -> tensor<6xi4>
    return
  }
}
{-# dialect_resources: { builtin: {
  // Alignment 1 followed by [-8, -7, -1, 0, 1, 7] and [-1, -1, -1].
  // CHECK-DAG: values: "0x0100000008090F000107"
  values: "0x0100000008090F000107",
  repeated: "0x010000000F0F0F"
  // CHECK-DAG: [[VALUES]]: "0x01000000F8F9FF000107"
  // CHECK-DAG: [[REPEATED]]: "0x01000000FFFFFF"
} } #-}

// -----

// Preserve sparse indices and implicit zeros while widening stored values.
// CHECK-LABEL: func.func @sparse_constant()
// CHECK-SAME: -> tensor<4xi64>
func.func @sparse_constant() -> tensor<4xi48> {
  // CHECK: %[[C:.*]] = tosa.const values(sparse<{{\[\[}}1], [3]], [-1, 140737488355327]> : tensor<4xi64>) : () -> tensor<4xi64>
  %0 = tosa.const values(sparse<[[1], [3]], [-1, 140737488355327]> : tensor<4xi48>) : () -> tensor<4xi48>
  // CHECK-NEXT: return %[[C]] : tensor<4xi64>
  return %0 : tensor<4xi48>
}

// -----

// Preserve floating-point values while widening i48 in the tensor encoding.
// CHECK-LABEL: func.func @float_constant_encoding_only()
// CHECK-SAME: -> tensor<2xf32, {i = i64}>
func.func @float_constant_encoding_only() -> tensor<2xf32, {i = i48}> {
  // CHECK: %[[C:.*]] = tosa.const values(dense<[-1.000000e+00, 7.000000e+00]> : tensor<2xf32, {i = i64}>) : () -> tensor<2xf32, {i = i64}>
  %0 = tosa.const values(dense<[-1.0, 7.0]> : tensor<2xf32, {i = i48}>) : () -> tensor<2xf32, {i = i48}>
  // CHECK-NEXT: return %[[C]] : tensor<2xf32, {i = i64}>
  return %0 : tensor<2xf32, {i = i48}>
}

// -----

// CHECK-LABEL: func.func @constants
func.func @constants() -> (tensor<5xi48>, tensor<2x3xi48>, tensor<i48>) {
  // CHECK: %[[C:.*]] = tosa.const values(dense<[-140737488355328, -1, 0, 1, 140737488355327]> : tensor<5xi64>) : () -> tensor<5xi64>
  %0 = tosa.const values(dense<[-140737488355328, -1, 0, 1, 140737488355327]> : tensor<5xi48>) : () -> tensor<5xi48>
  // CHECK: tosa.const values(dense<-1> : tensor<2x3xi64>) : () -> tensor<2x3xi64>
  %1 = tosa.const values(dense<-1> : tensor<2x3xi48>) : () -> tensor<2x3xi48>
  // CHECK: tosa.const values(dense<-140737488355328> : tensor<i64>) : () -> tensor<i64>
  %3 = tosa.const values(dense<-140737488355328> : tensor<i48>) : () -> tensor<i48>
  // CHECK: return %[[C]], %{{.*}}, %{{.*}} : tensor<5xi64>, tensor<2x3xi64>, tensor<i64>
  return %0, %1, %3 : tensor<5xi48>, tensor<2x3xi48>, tensor<i48>
}

// -----

// Widen packed dense arrays without reinterpreting their six-byte elements.
// Cover boundary values, empty arrays and repeated values that remain arrays.
// CHECK-LABEL: func.func @dense_arrays(
func.func @dense_arrays(%arg: tensor<2xi48>) -> tensor<2xi48> {
  // CHECK: tosa.identity %arg0
  // CHECK-SAME: test.empty = array<i64>
  // CHECK-SAME: test.repeated = array<i64: -1, -1, -1>
  // CHECK-SAME: test.values = array<i64: -140737488355328, -1, 0, 1, 2, 140737488355327>
  %0 = tosa.identity %arg {
    test.empty = array<i48>,
    test.repeated = array<i48: -1, -1, -1>,
    test.values = array<i48: -140737488355328, -1, 0, 1, 2, 140737488355327>
  } : (tensor<2xi48>) -> tensor<2xi48>
  return %0 : tensor<2xi48>
}

// -----

// Dense constant attributes and results must use the same widened encoding.
// CHECK-LABEL: func.func @encoded_constant()
// CHECK-SAME: -> tensor<2xi64, {i = i64, j = "k"}>
func.func @encoded_constant() -> tensor<2xi48, {i = i48, j = "k"}> {
  // CHECK: %[[C:.*]] = tosa.const values(dense<-1> : tensor<2xi64, {i = i64, j = "k"}>) : () -> tensor<2xi64, {i = i64, j = "k"}>
  %0 = tosa.const values(dense<-1> : tensor<2xi48, {i = i48, j = "k"}>) : () -> tensor<2xi48, {i = i48, j = "k"}>
  // CHECK-NEXT: return %[[C]] : tensor<2xi64, {i = i64, j = "k"}>
  return %0 : tensor<2xi48, {i = i48, j = "k"}>
}

// -----

// Widen each complete element in the resource without resolving its shape.
// CHECK-LABEL: func.func @dynamic_resource()
// CHECK-SAME: -> tensor<?xi64>
func.func @dynamic_resource() -> tensor<?xi48> {
  // CHECK: %[[C:.*]] = tosa.const values(dense_resource<[[DYNAMIC:[a-zA-Z0-9_]+]]> : tensor<?xi64>) : () -> tensor<?xi64>
  %0 = tosa.const values(dense_resource<dynamic> : tensor<?xi48>) : () -> tensor<?xi48>
  // CHECK-NEXT: return %[[C]] : tensor<?xi64>
  return %0 : tensor<?xi48>
}
{-#
  dialect_resources: {
    builtin: {
      // Alignment 1 followed by one little-endian i48 value of -1.
      // CHECK: [[DYNAMIC]]: "0x08000000FFFFFFFFFFFFFFFF"
      dynamic: "0x01000000FFFFFFFFFFFF"
    }
  }
#-}

// -----

// Check that dense constants and resource payloads remain correct after
// one or two runs of the pass.
// The module attribute and both functions must share the widened resource.
// An i8 view of the same blob must retain the original bytes.
// CHECK-LABEL: module attributes
// CHECK-SAME: {test.bytes = dense_resource<values> : tensor<30xi8>, test.original = dense_resource<[[VALUES:[a-zA-Z0-9_]+]]> : tensor<5xi64>}
module attributes {test.bytes = dense_resource<values> : tensor<30xi8>, test.original = dense_resource<values> : tensor<5xi48>} {
  // CHECK: func.func @resource_values
  func.func @resource_values() {
    // CHECK: tosa.const values(dense_resource<[[VALUES]]> : tensor<5xi64>) : () -> tensor<5xi64>
    %0 = tosa.const values(dense_resource<values> : tensor<5xi48>) : () -> tensor<5xi48>
    // CHECK: tosa.const values(dense<-1> : tensor<2xi64>) : () -> tensor<2xi64>
    %dense = tosa.const values(dense<-1> : tensor<2xi48>) : () -> tensor<2xi48>
    return
  }
  // CHECK: func.func @shared_resource
  func.func @shared_resource() {
    // CHECK: tosa.const values(dense_resource<[[VALUES]]> : tensor<5xi64>) : () -> tensor<5xi64>
    %0 = tosa.const values(dense_resource<values> : tensor<5xi48>) : () -> tensor<5xi48>
    return
  }
}

{-#
  dialect_resources: {
    builtin: {
      // Each blob starts with a four-byte little-endian alignment header.
      // 01000000 means byte alignment; the remaining bytes are packed i48s.
      // Values: [-140737488355328, -1, 0, 1, 140737488355327].
      // CHECK-DAG: values: "0x01000000000000000080FFFFFFFFFFFF000000000000010000000000FFFFFFFFFF7F"
      values: "0x01000000000000000080FFFFFFFFFFFF000000000000010000000000FFFFFFFFFF7F"
      // Widened blobs have alignment 8 (08000000) and the same numeric values
      // stored as eight-byte little-endian integers, sign-extended.
      // CHECK-DAG: [[VALUES]]: "0x08000000000000000080FFFFFFFFFFFFFFFFFFFF00000000000000000100000000000000FFFFFFFFFF7F0000"
    }
  }
#-}

// -----

// Unselected widths remain unchanged in payloads, including unresolved resources.
// CHECK: module @constants attributes {
// CHECK-SAME: test.array = array<i128: -1>
// CHECK-SAME: test.resource = dense_resource<missing> : tensor<2xi128>
// CHECK-SAME: test.scalar = -1 : i128
// CHECK-SAME: test.values = dense<[-4, -1, 0, 3]> : tensor<4xi3>
module @constants attributes {
  test.array = array<i128: -1>,
  test.resource = dense_resource<missing> : tensor<2xi128>,
  test.scalar = -1 : i128,
  test.values = dense<[-4, -1, 0, 3]> : tensor<4xi3>
} {}
