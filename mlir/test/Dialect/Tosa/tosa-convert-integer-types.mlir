// REQUIRES: host-byteorder-little-endian
// RUN: mlir-opt -split-input-file -pass-pipeline='builtin.module(tosa-convert-integer-types{bitwidth-mappings=4:8,48:64})' %s | FileCheck %s --implicit-check-not=i4 --implicit-check-not=unrealized_conversion_cast --implicit-check-not=tosa.cast
// RUN: mlir-opt -split-input-file -tosa-convert-integer-types='bitwidth-mappings=4:8,48:64' %s | mlir-opt -split-input-file -tosa-convert-integer-types='bitwidth-mappings=4:8,48:64' | FileCheck %s --implicit-check-not=i4 --implicit-check-not=unrealized_conversion_cast --implicit-check-not=tosa.cast

// CHECK-LABEL: func.func @identity(
// CHECK-SAME: %[[ARG:.*]]: tensor<?x2xi64>) -> tensor<?x2xi64>
func.func @identity(%arg: tensor<?x2xi48>) -> tensor<?x2xi48> {
  // CHECK: %[[ID:.*]] = tosa.identity %[[ARG]] {test.marker = "keep"} : (tensor<?x2xi64>) -> tensor<?x2xi64>
  %0 = tosa.identity %arg {test.marker = "keep"} : (tensor<?x2xi48>) -> tensor<?x2xi48>
  // CHECK-NEXT: return %[[ID]] : tensor<?x2xi64>
  return %0 : tensor<?x2xi48>
}

// -----

// CHECK-LABEL: func.func @clamp
func.func @clamp(%arg: tensor<4xi48>) -> tensor<4xi48> {
  // CHECK: tosa.clamp %{{.*}} min_val(-140737488355328 : i64) max_val(140737488355327 : i64) {test.marker = "keep"} : (tensor<4xi64>) -> tensor<4xi64>
  %0 = tosa.clamp %arg min_val(-140737488355328 : i48) max_val(140737488355327 : i48) {test.marker = "keep"} : (tensor<4xi48>) -> tensor<4xi48>
  return %0 : tensor<4xi48>
}

// -----

// CHECK-LABEL: func.func @reshape
func.func @reshape(%arg: tensor<2x3xi48>) -> tensor<6xi48> {
  // CHECK: %[[SHAPE:.*]] = tosa.const_shape values(dense<6> : tensor<1xindex>) : () -> !tosa.shape<1>
  %shape = tosa.const_shape values(dense<6> : tensor<1xindex>) : () -> !tosa.shape<1>
  // CHECK: tosa.reshape %{{.*}}, %[[SHAPE]] : (tensor<2x3xi64>, !tosa.shape<1>) -> tensor<6xi64>
  %0 = tosa.reshape %arg, %shape : (tensor<2x3xi48>, !tosa.shape<1>) -> tensor<6xi48>
  return %0 : tensor<6xi48>
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

// CHECK-LABEL: func.func @while_loop
func.func @while_loop(%arg: tensor<i48>, %limit: tensor<i48>) -> tensor<i48> {
  // CHECK: %[[LOOP:.*]] = tosa.while_loop
  // CHECK-SAME: (tensor<i64>) -> tensor<i64>
  %0 = tosa.while_loop (%iter = %arg) : (tensor<i48>) -> tensor<i48> {
    // CHECK: tosa.greater %{{.*}}, %{{.*}} : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %cond = tosa.greater %limit, %iter : (tensor<i48>, tensor<i48>) -> tensor<i1>
    // CHECK: tosa.yield %{{.*}} : tensor<i1>
    tosa.yield %cond : tensor<i1>
  } do {
  // CHECK: ^bb0(%[[ITER:.*]]: tensor<i64>):
  ^bb0(%iter: tensor<i48>):
    // CHECK: %[[ONE:.*]] = tosa.const values(dense<1> : tensor<i64>) : () -> tensor<i64>
    %one = tosa.const values(dense<1> : tensor<i48>) : () -> tensor<i48>
    // CHECK: %[[ADD:.*]] = tosa.add %[[ITER]], %[[ONE]] : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %next = tosa.add %iter, %one : (tensor<i48>, tensor<i48>) -> tensor<i48>
    // CHECK: tosa.yield %[[ADD]] : tensor<i64>
    tosa.yield %next : tensor<i48>
  }
  // CHECK: return %[[LOOP]] : tensor<i64>
  return %0 : tensor<i48>
}

// -----

// CHECK: func.func private @external(tensor<?xi64>) -> tensor<*xi64>
func.func private @external(tensor<?xi48>) -> tensor<*xi48>

// -----

// Widen function arguments even when the body does not use them.
// CHECK-LABEL: func.func @unused_arguments(
// CHECK-SAME: %{{.*}}: tensor<?x2xi64>, %{{.*}}: tensor<*xi64>, %{{.*}}: tensor<2xui64>)
func.func @unused_arguments(%ranked: tensor<?x2xi48>, %unranked: tensor<*xi48>, %unsigned: tensor<2xui48>) {
  // CHECK-NEXT: return
  return
}

// -----

// Exercise an accumulator attribute even when no operand or result needs widening.
// CHECK-LABEL: func.func @accumulator_only
func.func @accumulator_only(%input: tensor<1x3x3x1xi16>, %weight: tensor<1x1x1x1xi8>, %bias: tensor<1xi32>, %izp: tensor<1xi16>, %wzp: tensor<1xi8>) -> tensor<1x3x3x1xi32> {
  // CHECK: tosa.conv2d
  // CHECK-SAME: acc_type(i64)
  // CHECK-SAME: -> tensor<1x3x3x1xi32>
  %0 = tosa.conv2d %input, %weight, %bias, %izp, %wzp pad([0, 0, 0, 0]) stride([1, 1]) dilation([1, 1]) acc_type(i48) : (tensor<1x3x3x1xi16>, tensor<1x1x1x1xi8>, tensor<1xi32>, tensor<1xi16>, tensor<1xi8>) -> tensor<1x3x3x1xi32>
  return %0 : tensor<1x3x3x1xi32>
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

// CHECK-LABEL: func.func @casts
func.func @casts(%arg: tensor<2xi32>) -> tensor<2xi8> {
  // CHECK: %[[IN:.*]] = tosa.cast %arg0 input_unsigned(true) {test.marker = "keep"} : (tensor<2xi32>) -> tensor<2xi64>
  %0 = tosa.cast %arg input_unsigned(true) {test.marker = "keep"} : (tensor<2xi32>) -> tensor<2xi64>
  // CHECK: %[[OUT:.*]] = tosa.cast %[[IN]] : (tensor<2xi64>) -> tensor<2xi8>
  %1 = tosa.cast %0 : (tensor<2xi64>) -> tensor<2xi8>
  // CHECK: return %[[OUT]] : tensor<2xi8>
  return %1 : tensor<2xi8>
}

// -----

// CHECK-LABEL: func.func @unchanged(
// CHECK-SAME: %[[ARG:.*]]: tensor<2xi64>) -> (tensor<2xi64>, tensor<2xf32>, i32)
func.func @unchanged(%arg: tensor<2xi64>) -> (tensor<2xi64>, tensor<2xf32>, i32) {
  // CHECK: %[[ID:.*]] = tosa.identity %[[ARG]] : (tensor<2xi64>) -> tensor<2xi64>
  %0 = tosa.identity %arg : (tensor<2xi64>) -> tensor<2xi64>
  // CHECK: %[[C:.*]] = tosa.const values(dense<1.000000e+00> : tensor<2xf32>) : () -> tensor<2xf32>
  %1 = tosa.const values(dense<1.0> : tensor<2xf32>) : () -> tensor<2xf32>
  // CHECK: %[[A:.*]] = arith.constant 48 : i32
  %2 = arith.constant 48 : i32
  // CHECK-NEXT: return %[[ID]], %[[C]], %[[A]] : tensor<2xi64>, tensor<2xf32>, i32
  return %0, %1, %2 : tensor<2xi64>, tensor<2xf32>, i32
}

// -----

// Recurse into nested attributes even when operand and result types are unchanged.
// CHECK-LABEL: func.func @attributes_only
func.func @attributes_only(%arg: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK: tosa.identity %arg0 {test.nested = [{signed = -140737488355328 : si64, type = tensor<2xi64>, unsigned = 281474976710655 : ui64}], test.unchanged = 48 : i32} : (tensor<2xi32>) -> tensor<2xi32>
  %0 = tosa.identity %arg {test.nested = [{signed = -140737488355328 : si48, type = tensor<2xi48>, unsigned = 281474976710655 : ui48}], test.unchanged = 48 : i32} : (tensor<2xi32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// -----

// Widen nested values even when the enclosing operation's types are unchanged.
// CHECK-LABEL: func.func @internal_region(
// CHECK-SAME: %[[COND:.*]]: tensor<i1>) -> tensor<i1>
func.func @internal_region(%cond: tensor<i1>) -> tensor<i1> {
  // CHECK: %[[IF:.*]] = tosa.cond_if %[[COND]] : tensor<i1> -> tensor<i1>
  %0 = tosa.cond_if %cond : tensor<i1> -> tensor<i1> {
    // CHECK: %[[C:.*]] = tosa.const values(dense<1> : tensor<i64>) : () -> tensor<i64>
    %c = tosa.const values(dense<1> : tensor<i48>) : () -> tensor<i48>
    // CHECK: %[[D:.*]] = tosa.const values(dense<2> : tensor<i64>) : () -> tensor<i64>
    %d = tosa.const values(dense<2> : tensor<i48>) : () -> tensor<i48>
    // CHECK: %[[SELECT:.*]] = tosa.select %[[COND]], %[[C]], %[[D]] : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
    %select = tosa.select %cond, %c, %d : (tensor<i1>, tensor<i48>, tensor<i48>) -> tensor<i48>
    // CHECK: %[[EQ:.*]] = tosa.equal %[[SELECT]], %[[C]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %eq = tosa.equal %select, %c : (tensor<i48>, tensor<i48>) -> tensor<i1>
    // CHECK: tosa.yield %[[EQ]] : tensor<i1>
    tosa.yield %eq : tensor<i1>
  } else {
    // CHECK: tosa.yield %[[COND]] : tensor<i1>
    tosa.yield %cond : tensor<i1>
  }
  // CHECK: return %[[IF]] : tensor<i1>
  return %0 : tensor<i1>
}

// -----

// Mixed types and multiple results; preserve function and call attributes.
// CHECK-LABEL: func.func private @mixed(
// CHECK-SAME: %[[A:.*]]: tensor<?xi64>, %[[B:.*]]: tensor<*xui64>, %[[C:.*]]: tensor<2xi32>) -> (tensor<*xui64>, tensor<?xi64>, tensor<2xi32>)
func.func private @mixed(%a: tensor<?xi48>, %b: tensor<*xui48>, %c: tensor<2xi32>) -> (tensor<*xui48>, tensor<?xi48>, tensor<2xi32>) {
  // CHECK: return %[[B]], %[[A]], %[[C]] : tensor<*xui64>, tensor<?xi64>, tensor<2xi32>
  return %b, %a, %c : tensor<*xui48>, tensor<?xi48>, tensor<2xi32>
}
// CHECK-LABEL: func.func @mixed_caller(
// CHECK-SAME: %[[A:.*]]: tensor<?xi64>, %[[B:.*]]: tensor<*xui64>, %[[C:.*]]: tensor<2xi32>) -> (tensor<*xui64>, tensor<?xi64>, tensor<2xi32>) attributes {test.marker = "function"}
func.func @mixed_caller(%a: tensor<?xi48>, %b: tensor<*xui48>, %c: tensor<2xi32>) -> (tensor<*xui48>, tensor<?xi48>, tensor<2xi32>) attributes {test.marker = "function"} {
  // CHECK: %[[CALL:.*]]:3 = call @mixed(%[[A]], %[[B]], %[[C]]) {test.marker = "call"} : (tensor<?xi64>, tensor<*xui64>, tensor<2xi32>) -> (tensor<*xui64>, tensor<?xi64>, tensor<2xi32>)
  %0:3 = func.call @mixed(%a, %b, %c) {test.marker = "call"} : (tensor<?xi48>, tensor<*xui48>, tensor<2xi32>) -> (tensor<*xui48>, tensor<?xi48>, tensor<2xi32>)
  // CHECK: return %[[CALL]]#0, %[[CALL]]#1, %[[CALL]]#2 : tensor<*xui64>, tensor<?xi64>, tensor<2xi32>
  return %0#0, %0#1, %0#2 : tensor<*xui48>, tensor<?xi48>, tensor<2xi32>
}

// -----

// Convert module metadata, including constant payloads without SSA users.
// CHECK-LABEL: module @module_attributes
// CHECK-SAME: test.constant = dense<[-1, 140737488355327]> : tensor<2xi64>
// CHECK-SAME: test.type = tensor<2xi64>
// CHECK-SAME: test.value = -1 : i64
module @module_attributes attributes {
  test.constant = dense<[-1, 140737488355327]> : tensor<2xi48>,
  test.type = tensor<2xi48>,
  test.value = -1 : i48
} {}

// -----

// Convert function, argument and result metadata along with the signature.
// CHECK-LABEL: func.func @function_attributes(
// CHECK-SAME: %[[ARG:.*]]: tensor<2xi64> {test.type = tensor<2xi64>}) -> (tensor<2xi64> {test.type = tensor<2xi64>}) attributes {test.value = -1 : i64}
func.func @function_attributes(%arg: tensor<2xi48> {test.type = tensor<2xi48>}) -> (tensor<2xi48> {test.type = tensor<2xi48>}) attributes {test.value = -1 : i48} {
  // CHECK: return %[[ARG]] : tensor<2xi64>
  return %arg : tensor<2xi48>
}

// -----

// Convert types embedded in tensor encodings while preserving other metadata.
// CHECK-LABEL: func.func @encoded_tensor(
// CHECK-SAME: %[[ARG:.*]]: tensor<?x2xi64, {i = i64, j = "k"}>) -> tensor<?x2xi64, {i = i64, j = "k"}>
func.func @encoded_tensor(%arg: tensor<?x2xi48, {i = i48, j = "k"}>) -> tensor<?x2xi48, {i = i48, j = "k"}> {
  // CHECK: return %[[ARG]] : tensor<?x2xi64, {i = i64, j = "k"}>
  return %arg : tensor<?x2xi48, {i = i48, j = "k"}>
}

// -----

// Preserve stride metadata on tensor arguments and results while widening.
// CHECK-LABEL: func.func @strided_tensor(
// CHECK-SAME: %[[ARG:.*]]: tensor<2x3xi64> {test.strides = strided<[5, 1]>}) -> (tensor<2x3xi64> {test.strides = strided<[5, 1]>})
func.func @strided_tensor(%arg: tensor<2x3xi48> {test.strides = strided<[5, 1]>}) -> (tensor<2x3xi48> {test.strides = strided<[5, 1]>}) {
  // CHECK: return %[[ARG]] : tensor<2x3xi64>
  return %arg : tensor<2x3xi48>
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

// Convert a function reference, its indirect call and a function-valued result.
// CHECK: func.func private @referenced(tensor<2xi64>) -> tensor<2xi64>
func.func private @referenced(tensor<2xi48>) -> tensor<2xi48>
// CHECK-LABEL: func.func @function_reference(
// CHECK-SAME: %[[ARG:.*]]: tensor<2xi64>) -> (tensor<2xi64>, (tensor<2xi64>) -> tensor<2xi64>)
func.func @function_reference(%arg: tensor<2xi48>) -> (tensor<2xi48>, (tensor<2xi48>) -> tensor<2xi48>) {
  // CHECK: %[[FN:.*]] = constant @referenced : (tensor<2xi64>) -> tensor<2xi64>
  %fn = func.constant @referenced : (tensor<2xi48>) -> tensor<2xi48>
  // CHECK: %[[RESULT:.*]] = call_indirect %[[FN]](%[[ARG]]) : (tensor<2xi64>) -> tensor<2xi64>
  %result = func.call_indirect %fn(%arg) : (tensor<2xi48>) -> tensor<2xi48>
  // CHECK: return %[[RESULT]], %[[FN]] : tensor<2xi64>, (tensor<2xi64>) -> tensor<2xi64>
  return %result, %fn : tensor<2xi48>, (tensor<2xi48>) -> tensor<2xi48>
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
// CHECK: func.func private @types(i1, i3, i8, i7, i8, i9, i16, i24, i32, i64, i64, i128) -> (si3, ui3)
func.func private @types(i1, i3, i4, i7, i8, i9, i16, i24, i32, i48, i64, i128) -> (si3, ui3)

// -----

// Constant payloads and resource storage.

// CHECK-LABEL: func.func @subbyte_constants()
// CHECK-SAME: -> (tensor<5xi8>, tensor<2x3xi8>, tensor<3xui8>)
func.func @subbyte_constants() -> (tensor<5xi4>, tensor<2x3xi4>, tensor<3xui4>) {
  // CHECK: tosa.const values(dense<[-7, -1, 0, 1, 7]> : tensor<5xi8>)
  %a = tosa.const values(dense<[-7, -1, 0, 1, 7]> : tensor<5xi4>) : () -> tensor<5xi4>
  // CHECK: tosa.const values(dense<-7> : tensor<2x3xi8>)
  %b = tosa.const values(dense<-7> : tensor<2x3xi4>) : () -> tensor<2x3xi4>
  // CHECK: tosa.const values(dense<[0, 8, 15]> : tensor<3xui8>)
  %c = tosa.const values(dense<[0, 8, 15]> : tensor<3xui4>) : () -> tensor<3xui4>
  return %a, %b, %c : tensor<5xi4>, tensor<2x3xi4>, tensor<3xui4>
}

// -----

// Widen scalar and dense attributes, including explicitly signed types.
// CHECK-LABEL: func.func @attributes(
// CHECK-SAME: tensor<2xi8, {type = i8}>) -> tensor<2xi8, {type = i8}>
func.func @attributes(%arg: tensor<2xi4, {type = i4}>) -> tensor<2xi4, {type = i4}> {
  // CHECK: tosa.identity %arg0
  // CHECK-SAME: test.nested = [{signed = -7 : si8, signless = -1 : i8, unsigned = 15 : ui8}]
  // CHECK-SAME: test.signed = dense<[-7, -1, 7]> : tensor<3xsi8>
  %0 = tosa.identity %arg {
    test.nested = [{signed = -7 : si4, signless = -1 : i4, unsigned = 15 : ui4}],
    test.signed = dense<[-7, -1, 7]> : tensor<3xsi4>
  } : (tensor<2xi4, {type = i4}>) -> tensor<2xi4, {type = i4}>
  return %0 : tensor<2xi4, {type = i4}>
}

// -----

// Widen sparse i4 values while preserving their coordinates.
// CHECK-LABEL: func.func @subbyte_sparse_constant()
// CHECK-SAME: -> tensor<4xi8>
func.func @subbyte_sparse_constant() -> tensor<4xi4> {
  // CHECK: tosa.const values(sparse<{{\[\[}}1], [3]], [-7, 7]> : tensor<4xi8>)
  %0 = tosa.const values(sparse<[[1], [3]], [-7, 7]> : tensor<4xi4>) : () -> tensor<4xi4>
  return %0 : tensor<4xi4>
}

// -----

// i4 resources use one byte per element. Preserve the original blob for an i8
// view, share the widened resource, and retain storage for repeated values.
// CHECK: module attributes {test.bytes = dense_resource<signed_data> : tensor<5xi8>, test.original = dense_resource<[[SIGNED:[a-zA-Z0-9_]+]]> : tensor<5xi8>}
module attributes {
  test.bytes = dense_resource<signed_data> : tensor<5xi8>,
  test.original = dense_resource<signed_data> : tensor<5xi4>
} {
  // CHECK: func.func @resources()
  // CHECK-SAME: attributes {test.signed = dense_resource<[[REPEATED:[a-zA-Z0-9_]+]]> : tensor<3xsi8>}
  func.func @resources() attributes {test.signed = dense_resource<repeated> : tensor<3xsi4>} {
    // CHECK: tosa.const values(dense_resource<[[SIGNED]]> : tensor<5xi8>)
    %a = tosa.const values(dense_resource<signed_data> : tensor<5xi4>) : () -> tensor<5xi4>
    // CHECK: tosa.const values(dense_resource<[[UNSIGNED:[a-zA-Z0-9_]+]]> : tensor<3xui8>)
    %b = tosa.const values(dense_resource<unsigned_data> : tensor<3xui4>) : () -> tensor<3xui4>
    return
  }
  // CHECK: func.func @shared_resource()
  func.func @shared_resource() {
    // CHECK: tosa.const values(dense_resource<[[SIGNED]]> : tensor<5xi8>)
    %0 = tosa.const values(dense_resource<signed_data> : tensor<5xi4>) : () -> tensor<5xi4>
    return
  }
}
{-# dialect_resources: { builtin: {
  // Alignment 1 followed by [-7, -1, 0, 1, 7], [0, 8, 15], and [-1, -1, -1].
  // CHECK-DAG: signed_data: "0x01000000090F000107"
  signed_data: "0x01000000090F000107",
  unsigned_data: "0x0100000000080F",
  repeated: "0x010000000F0F0F"
  // CHECK-DAG: [[SIGNED]]: "0x01000000F9FF000107"
  // CHECK-DAG: [[UNSIGNED]]: "0x0100000000080F"
  // CHECK-DAG: [[REPEATED]]: "0x01000000FFFFFF"
} } #-}

// -----

// CHECK-LABEL: func.func @subbyte_dynamic_resource()
// CHECK-SAME: -> tensor<?xi8>
func.func @subbyte_dynamic_resource() -> tensor<?xi4> {
  // CHECK: tosa.const values(dense_resource<[[DYNAMIC:[a-zA-Z0-9_]+]]> : tensor<?xi8>)
  %0 = tosa.const values(dense_resource<dynamic> : tensor<?xi4>) : () -> tensor<?xi4>
  return %0 : tensor<?xi4>
}
{-# dialect_resources: { builtin: {
  dynamic: "0x01000000090F"
  // CHECK: [[DYNAMIC]]: "0x01000000F9FF"
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

// CHECK-LABEL: func.func @unsigned_constants
func.func @unsigned_constants() -> tensor<3xui48> {
  // CHECK: tosa.const values(dense<[0, 140737488355328, 281474976710655]> : tensor<3xui64>) : () -> tensor<3xui64>
  %0 = tosa.const values(dense<[0, 140737488355328, 281474976710655]> : tensor<3xui48>) : () -> tensor<3xui48>
  return %0 : tensor<3xui48>
}

// -----

// Widen packed dense arrays without reinterpreting their six-byte elements.
// Cover signedness, empty arrays and repeated values that must remain arrays.
// CHECK-LABEL: func.func @dense_arrays(
func.func @dense_arrays(%arg: tensor<2xi48>) -> tensor<2xi48> {
  // CHECK: tosa.identity %arg0
  // CHECK-SAME: test.empty = array<i64>
  // CHECK-SAME: test.repeated = array<i64: -1, -1, -1>
  // CHECK-SAME: test.signed = array<si64: -140737488355328, -1, 140737488355327>
  // CHECK-SAME: test.signless = array<i64: 1, -1, 2>
  // CHECK-SAME: test.unsigned = array<ui64: 0, 140737488355328, 281474976710655>
  %0 = tosa.identity %arg {
    test.empty = array<i48>,
    test.repeated = array<i48: -1, -1, -1>,
    test.signed = array<si48: -140737488355328, -1, 140737488355327>,
    test.signless = array<i48: 1, -1, 2>,
    test.unsigned = array<ui48: 0, 140737488355328, 281474976710655>
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

// Convert dense constant encodings even when the element type is unchanged.
// CHECK-LABEL: func.func @encoded_constant_encoding_only()
// CHECK-SAME: -> tensor<2xi32, {i = i64, j = "k"}>
func.func @encoded_constant_encoding_only() -> tensor<2xi32, {i = i48, j = "k"}> {
  // CHECK: %[[C:.*]] = tosa.const values(dense<[-1, 7]> : tensor<2xi32, {i = i64, j = "k"}>) : () -> tensor<2xi32, {i = i64, j = "k"}>
  %0 = tosa.const values(dense<[-1, 7]> : tensor<2xi32, {i = i48, j = "k"}>) : () -> tensor<2xi32, {i = i48, j = "k"}>
  // CHECK-NEXT: return %[[C]] : tensor<2xi32, {i = i64, j = "k"}>
  return %0 : tensor<2xi32, {i = i48, j = "k"}>
}

// -----

// Preserve layout metadata and element order while widening a dense constant.
// CHECK-LABEL: func.func @strided_constant()
// CHECK-SAME: -> tensor<2x2xi64>
func.func @strided_constant() -> tensor<2x2xi48> {
  // CHECK: %[[C:.*]] = tosa.const values(dense<{{\[}}[-140737488355328, -1], [0, 140737488355327]]> : tensor<2x2xi64>) {test.layout = strided<[1, 2]>} : () -> tensor<2x2xi64>
  %0 = tosa.const values(dense<[[-140737488355328, -1], [0, 140737488355327]]> : tensor<2x2xi48>) {test.layout = strided<[1, 2]>} : () -> tensor<2x2xi48>
  // CHECK-NEXT: return %[[C]] : tensor<2x2xi64>
  return %0 : tensor<2x2xi48>
}

// -----

// Preserve layout metadata and element order while widening a resource payload.
// CHECK-LABEL: func.func @strided_resource_constant()
// CHECK-SAME: -> tensor<2x2xi64>
func.func @strided_resource_constant() -> tensor<2x2xi48> {
  // CHECK: %[[C:.*]] = tosa.const values(dense_resource<[[STRIDED:[a-zA-Z0-9_]+]]> : tensor<2x2xi64>) {test.layout = strided<[1, 2]>} : () -> tensor<2x2xi64>
  %0 = tosa.const values(dense_resource<strided> : tensor<2x2xi48>) {test.layout = strided<[1, 2]>} : () -> tensor<2x2xi48>
  // CHECK-NEXT: return %[[C]] : tensor<2x2xi64>
  return %0 : tensor<2x2xi48>
}

{-# dialect_resources: { builtin: {
  // Signed i48 minimum, -1, zero and maximum, sign-extended to i64 in order.
  // CHECK: [[STRIDED]]: "0x08000000000000000080FFFFFFFFFFFFFFFFFFFF0000000000000000FFFFFFFFFF7F0000"
  strided: "0x01000000000000000080FFFFFFFFFFFF000000000000FFFFFFFFFF7F"
} } #-}

// -----

// Repeated values must retain storage for every element in the resource.
// CHECK-LABEL: func.func @repeated_resource()
func.func @repeated_resource() -> tensor<2x2xi48> {
  // CHECK: tosa.const values(dense_resource<[[REPEATED:[a-zA-Z0-9_]+]]> : tensor<2x2xi64>)
  %0 = tosa.const values(dense_resource<repeated> : tensor<2x2xi48>) : () -> tensor<2x2xi48>
  return %0 : tensor<2x2xi48>
}

{-# dialect_resources: { builtin: {
  // Four copies of -1, each occupying six bytes in the input and eight in the output.
  // CHECK: [[REPEATED]]: "0x08000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"
  repeated: "0x01000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"
} } #-}

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

// Unranked tensors are valid in the dialect before shape inference. Widening
// the resource's elements should preserve its unranked shape.
// CHECK-LABEL: func.func @unranked_resource()
// CHECK-SAME: -> tensor<*xi64>
func.func @unranked_resource() -> tensor<*xi48> {
  // CHECK: %[[C:.*]] = tosa.const values(dense_resource<[[UNRANKED:[a-zA-Z0-9_]+]]> : tensor<*xi64>) : () -> tensor<*xi64>
  %0 = tosa.const values(dense_resource<unranked> : tensor<*xi48>) : () -> tensor<*xi48>
  // CHECK-NEXT: return %[[C]] : tensor<*xi64>
  return %0 : tensor<*xi48>
}
{-#
  dialect_resources: {
    builtin: {
      // Alignment 1 followed by one little-endian i48 value of -1.
      // CHECK: [[UNRANKED]]: "0x08000000FFFFFFFFFFFFFFFF"
      unranked: "0x01000000FFFFFFFFFFFF"
    }
  }
#-}

// -----

// Check that dense constants and resource payloads remain correct after
// one or two runs of the pass.
// The module attribute and both functions must share the widened resource.
// An i8 view of the same blob must retain the original bytes.
// CHECK-LABEL: module attributes
// CHECK-SAME: {test.bytes = dense_resource<signed_data> : tensor<30xi8>, test.original = dense_resource<[[SIGNED:[a-zA-Z0-9_]+]]> : tensor<5xi64>}
module attributes {test.bytes = dense_resource<signed_data> : tensor<30xi8>, test.original = dense_resource<signed_data> : tensor<5xi48>} {
  // CHECK: func.func @signed_resource
  func.func @signed_resource() {
    // CHECK: tosa.const values(dense_resource<[[SIGNED]]> : tensor<5xi64>) : () -> tensor<5xi64>
    %0 = tosa.const values(dense_resource<signed_data> : tensor<5xi48>) : () -> tensor<5xi48>
    // CHECK: tosa.const values(dense<-1> : tensor<2xi64>) : () -> tensor<2xi64>
    %dense = tosa.const values(dense<-1> : tensor<2xi48>) : () -> tensor<2xi48>
    return
  }
  // CHECK: func.func @shared_resource
  func.func @shared_resource() {
    // CHECK: tosa.const values(dense_resource<[[SIGNED]]> : tensor<5xi64>) : () -> tensor<5xi64>
    %0 = tosa.const values(dense_resource<signed_data> : tensor<5xi48>) : () -> tensor<5xi48>
    return
  }
  // CHECK: func.func @unsigned_resource
  func.func @unsigned_resource() {
    // CHECK: tosa.const values(dense_resource<[[UNSIGNED:[a-zA-Z0-9_]+]]> : tensor<3xui64>) : () -> tensor<3xui64>
    %0 = tosa.const values(dense_resource<unsigned_data> : tensor<3xui48>) : () -> tensor<3xui48>
    return
  }
}

{-#
  dialect_resources: {
    builtin: {
      // Each blob starts with a four-byte little-endian alignment header.
      // 01000000 means byte alignment; the remaining bytes are packed i48s.
      // Signed values: [-140737488355328, -1, 0, 1, 140737488355327].
      // CHECK-DAG: signed_data: "0x01000000000000000080FFFFFFFFFFFF000000000000010000000000FFFFFFFFFF7F"
      signed_data: "0x01000000000000000080FFFFFFFFFFFF000000000000010000000000FFFFFFFFFF7F",
      // Unsigned values: [0, 140737488355328, 281474976710655].
      unsigned_data: "0x01000000000000000000000000000080FFFFFFFFFFFF"
      // Widened blobs have alignment 8 (08000000) and the same numeric values
      // stored as eight-byte little-endian integers, sign- or zero-extended.
      // CHECK-DAG: [[SIGNED]]: "0x08000000000000000080FFFFFFFFFFFFFFFFFFFF00000000000000000100000000000000FFFFFFFFFF7F0000"
      // CHECK-DAG: [[UNSIGNED]]: "0x0800000000000000000000000000000000800000FFFFFFFFFFFF0000"
    }
  }
#-}
// -----

// Unselected widths remain unchanged in payloads, including unresolved resources.
// CHECK: module @constants attributes {
// CHECK-SAME: test.array = array<i128: -1>
// CHECK-SAME: test.resource = dense_resource<missing> : tensor<2xi128>
// CHECK-SAME: test.scalar = -1 : i128
// CHECK-SAME: test.signed = dense<[-4, -1, 0, 3]> : tensor<4xi3>
// CHECK-SAME: test.unsigned = dense<[0, 4, 7]> : tensor<3xui3>
module @constants attributes {
  test.array = array<i128: -1>,
  test.resource = dense_resource<missing> : tensor<2xi128>,
  test.scalar = -1 : i128,
  test.signed = dense<[-4, -1, 0, 3]> : tensor<4xi3>,
  test.unsigned = dense<[0, 4, 7]> : tensor<3xui3>
} {}
