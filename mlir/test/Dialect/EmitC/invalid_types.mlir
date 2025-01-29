// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func.func @illegal_opaque_type_1() {
    // expected-error @+1 {{expected non empty string in !emitc.opaque type}}
    %1 = "emitc.variable"(){value = "42" : !emitc.opaque<"">} : () -> !emitc.opaque<"mytype">
}

// -----

func.func @illegal_opaque_type_2() {
    // expected-error @+1 {{pointer not allowed as outer type with !emitc.opaque, use !emitc.ptr instead}}
    %1 = "emitc.variable"(){value = "nullptr" : !emitc.opaque<"int32_t*">} : () -> !emitc.opaque<"int32_t*">
}

// -----

func.func @illegal_array_missing_spec(
    // expected-error @+1 {{expected non-function type}}
    %arg0: !emitc.array<>) {
}

// -----

func.func @illegal_array_missing_shape(
    // expected-error @+1 {{shape must not be empty}}
    %arg9: !emitc.array<i32>) {
}

// -----

func.func @illegal_array_missing_x(
    // expected-error @+1 {{expected 'x' in dimension list}}
    %arg0: !emitc.array<10>
) {
}

// -----

func.func @illegal_array_missing_type(
    // expected-error @+1 {{expected non-function type}}
    %arg0: !emitc.array<10x>
) {
}

// -----

func.func @illegal_array_dynamic_shape(
    // expected-error @+1 {{expected static shape}}
    %arg0: !emitc.array<10x?xi32>
) {
}

// -----

func.func @illegal_array_unranked(
    // expected-error @+1 {{expected non-function type}}
    %arg0: !emitc.array<*xi32>
) {
}

// -----

func.func @illegal_array_with_array_element_type(
    // expected-error @+1 {{invalid array element type}}
    %arg0: !emitc.array<4x!emitc.array<4xi32>>
) {
}

// -----

func.func @illegal_array_with_tensor_element_type(
    // expected-error @+1 {{invalid array element type}}
    %arg0: !emitc.array<4xtensor<4xi32>>
) {
}

// -----

func.func @illegal_array_with_lvalue_element_type(
    // expected-error @+1 {{invalid array element type}}
    %arg0: !emitc.array<4x!emitc.lvalue<i32>>
) {
}

// -----

func.func @illegal_integer_type(%arg0: i11, %arg1: i11) -> i11 {
    // expected-error @+1 {{'emitc.mul' op operand #0 must be floating-point type supported by EmitC or integer, index or opaque type supported by EmitC, but got 'i11'}}
    %mul = "emitc.mul" (%arg0, %arg1) : (i11, i11) -> i11
    return
}

// -----

func.func @illegal_float_type(%arg0: f80, %arg1: f80) {
    // expected-error @+1 {{'emitc.mul' op operand #0 must be floating-point type supported by EmitC or integer, index or opaque type supported by EmitC, but got 'f80'}}
    %mul = "emitc.mul" (%arg0, %arg1) : (f80, f80) -> f80
    return
}

// -----

func.func @illegal_lvalue_type_1() {
    // expected-error @+1 {{!emitc.lvalue cannot wrap !emitc.array type}}
    %v = "emitc.variable"(){value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.array<1xi32>>
    return
}

// -----

func.func @illegal_lvalue_type_2() {
    // expected-error @+1 {{!emitc.lvalue must wrap supported emitc type, but got '!emitc.lvalue<i32>'}}
    %v = "emitc.variable"(){value = #emitc.opaque<"">} : () -> !emitc.lvalue<!emitc.lvalue<i32>>
    return
}

// -----

func.func @illegal_lvalue_type_3() {
    // expected-error @+1 {{!emitc.lvalue must wrap supported emitc type, but got 'i17'}}
    %v = "emitc.variable"(){value = #emitc.opaque<"">} : () -> !emitc.lvalue<i17>
    return
}

// -----

func.func @illegal_pointee_type_1() {
    // expected-error @+1 {{'emitc.constant' op result #0 must be type supported by EmitC, but got '!emitc.ptr<i11>'}}
    %v = "emitc.constant"(){value = #emitc.opaque<"{}">} : () -> !emitc.ptr<i11>
    return
}

// -----

func.func @illegal_pointee_type_2() {
    // expected-error @+1 {{pointers to lvalues are not allowed}}
    %v = "emitc.constant"(){value = #emitc.opaque<"NULL">} : () -> !emitc.ptr<!emitc.lvalue<i32>>
    return
}

// -----

func.func @illegal_non_static_tensor_shape_type() {
    // expected-error @+1 {{'emitc.constant' op result #0 must be type supported by EmitC, but got 'tensor<?xf32>'}}
    %v = "emitc.constant"(){value = #emitc.opaque<"{}">} : () -> tensor<?xf32>
    return
}

// -----

func.func @illegal_tensor_array_element_type() {
    // expected-error @+1 {{'emitc.constant' op result #0 must be type supported by EmitC, but got 'tensor<!emitc.array<9xi16>>'}}
    %v = "emitc.constant"(){value = #emitc.opaque<"{}">} : () -> tensor<!emitc.array<9xi16>>
    return
}

// -----

func.func @illegal_tensor_integer_element_type() {
    // expected-error @+1 {{'emitc.constant' op result #0 must be type supported by EmitC, but got 'tensor<9xi11>'}}
    %v = "emitc.constant"(){value = #emitc.opaque<"{}">} : () -> tensor<9xi11>
    return
}

// -----

func.func @illegal_tuple_array_element_type() {
    // expected-error @+1 {{'emitc.constant' op result #0 must be type supported by EmitC, but got 'tuple<!emitc.array<9xf32>, f32>'}}
    %v = "emitc.constant"(){value = #emitc.opaque<"{}">} : () -> tuple<!emitc.array<9xf32>, f32>
    return
}

// -----

func.func @illegal_tuple_float_element_type() {
    // expected-error @+1 {{'emitc.constant' op result #0 must be type supported by EmitC, but got 'tuple<i32, f80>'}}
    %v = "emitc.constant"(){value = #emitc.opaque<"{}">} : () -> tuple<i32, f80>
    return
}
