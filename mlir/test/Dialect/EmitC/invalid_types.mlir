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

func.func @illegal_array_non_positive_dimenson(
    // expected-error @+1 {{dimensions must have positive size}}
    %arg0: !emitc.array<0xi32>
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

func.func @illegal_integer_type(%arg0: i11, %arg1: i11) -> i11 {
    // expected-error @+1 {{'emitc.mul' op operand #0 must be floating-point type supported by EmitC or integer type supported by EmitC or index or EmitC opaque type, but got 'i11'}}
    %mul = "emitc.mul" (%arg0, %arg1) : (i11, i11) -> i11
    return
}

// -----

func.func @illegal_float_type(%arg0: f80, %arg1: f80) {
    // expected-error @+1 {{'emitc.mul' op operand #0 must be floating-point type supported by EmitC or integer type supported by EmitC or index or EmitC opaque type, but got 'f80'}}
    %mul = "emitc.mul" (%arg0, %arg1) : (f80, f80) -> f80
    return
}
