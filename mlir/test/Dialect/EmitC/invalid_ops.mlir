// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func @const_attribute_return_type_1() {
    // expected-error @+1 {{'emitc.constant' op requires attribute's type ('i64') to match op's return type ('i32')}}
    %c0 = "emitc.constant"(){value = 42: i64} : () -> i32
    return
}

// -----

func @const_attribute_return_type_2() {
    // expected-error @+1 {{'emitc.constant' op requires attribute's type ('!emitc.opaque<"char">') to match op's return type ('!emitc.opaque<"mychar">')}}
    %c0 = "emitc.constant"(){value = "CHAR_MIN" : !emitc.opaque<"char">} : () -> !emitc.opaque<"mychar">
    return
}

// -----

func @index_args_out_of_range_1() {
    // expected-error @+1 {{'emitc.call' op index argument is out of range}}
    emitc.call "test" () {args = [0 : index]} : () -> ()
    return
}

// -----

func @index_args_out_of_range_2(%arg : i32) {
    // expected-error @+1 {{'emitc.call' op index argument is out of range}}
    emitc.call "test" (%arg, %arg) {args = [2 : index]} : (i32, i32) -> ()
    return
}

// -----

func @empty_callee() {
    // expected-error @+1 {{'emitc.call' op callee must not be empty}}
    emitc.call "" () : () -> ()
    return
}

// -----

func @nonetype_arg(%arg : i32) {
    // expected-error @+1 {{'emitc.call' op array argument has no type}}
    emitc.call "nonetype_arg"(%arg) {args = [0 : index, [0, 1, 2]]} : (i32) -> i32
    return
}

// -----

func @array_template_arg(%arg : i32) {
    // expected-error @+1 {{'emitc.call' op template argument has invalid type}}
    emitc.call "nonetype_template_arg"(%arg) {template_args = [[0, 1, 2]]} : (i32) -> i32
    return
}

// -----

func @dense_template_argument(%arg : i32) {
    // expected-error @+1 {{'emitc.call' op template argument has invalid type}}
    emitc.call "dense_template_argument"(%arg) {template_args = [dense<[1.0, 1.0]> : tensor<2xf32>]} : (i32) -> i32
    return
}

// -----

func @empty_operator(%arg : i32) {
    // expected-error @+1 {{'emitc.apply' op applicable operator must not be empty}}
    %2 = emitc.apply ""(%arg) : (i32) -> !emitc.ptr<i32>
    return
}

// -----

func @illegal_operator(%arg : i32) {
    // expected-error @+1 {{'emitc.apply' op applicable operator is illegal}}
    %2 = emitc.apply "+"(%arg) : (i32) -> !emitc.ptr<i32>
    return
}

// -----

func @var_attribute_return_type_1() {
    // expected-error @+1 {{'emitc.variable' op requires attribute's type ('i64') to match op's return type ('i32')}}
    %c0 = "emitc.variable"(){value = 42: i64} : () -> i32
    return
}

// -----

func @var_attribute_return_type_2() {
    // expected-error @+1 {{'emitc.variable' op requires attribute's type ('!emitc.ptr<i64>') to match op's return type ('!emitc.ptr<i32>')}}
    %c0 = "emitc.variable"(){value = "nullptr" : !emitc.ptr<i64>} : () -> !emitc.ptr<i32>
    return
}
