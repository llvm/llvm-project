// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func.func @const_attribute_str() {
    // expected-error @+1 {{'emitc.constant' op string attributes are not supported, use #emitc.opaque instead}}                 
    %c0 = "emitc.constant"(){value = "NULL"} : () -> !emitc.ptr<i32>
    return
}

// -----

func.func @const_attribute_return_type_1() {
    // expected-error @+1 {{'emitc.constant' op requires attribute to either be an #emitc.opaque attribute or it's type ('i64') to match the op's result type ('i32')}}
    %c0 = "emitc.constant"(){value = 42: i64} : () -> i32
    return
}

// -----

func.func @const_attribute_return_type_2() {
    // expected-error @+1 {{'emitc.constant' op attribute 'value' failed to satisfy constraint: An opaque attribute or TypedAttr instance}}
    %c0 = "emitc.constant"(){value = unit} : () -> i32
    return
}

// -----

func.func @empty_constant() {
    // expected-error @+1 {{'emitc.constant' op value must not be empty}}
    %c0 = "emitc.constant"(){value = #emitc.opaque<"">} : () -> i32
    return
}

// -----

func.func @index_args_out_of_range_1() {
    // expected-error @+1 {{'emitc.call_opaque' op index argument is out of range}}
    emitc.call_opaque "test" () {args = [0 : index]} : () -> ()
    return
}

// -----

func.func @index_args_out_of_range_2(%arg : i32) {
    // expected-error @+1 {{'emitc.call_opaque' op index argument is out of range}}
    emitc.call_opaque "test" (%arg, %arg) {args = [2 : index]} : (i32, i32) -> ()
    return
}

// -----

func.func @empty_callee() {
    // expected-error @+1 {{'emitc.call_opaque' op callee must not be empty}}
    emitc.call_opaque "" () : () -> ()
    return
}

// -----

func.func @nonetype_arg(%arg : i32) {
    // expected-error @+1 {{'emitc.call_opaque' op array argument has no type}}
    emitc.call_opaque "nonetype_arg"(%arg) {args = [0 : index, [0, 1, 2]]} : (i32) -> i32
    return
}

// -----

func.func @array_template_arg(%arg : i32) {
    // expected-error @+1 {{'emitc.call_opaque' op template argument has invalid type}}
    emitc.call_opaque "nonetype_template_arg"(%arg) {template_args = [[0, 1, 2]]} : (i32) -> i32
    return
}

// -----

func.func @dense_template_argument(%arg : i32) {
    // expected-error @+1 {{'emitc.call_opaque' op template argument has invalid type}}
    emitc.call_opaque "dense_template_argument"(%arg) {template_args = [dense<[1.0, 1.0]> : tensor<2xf32>]} : (i32) -> i32
    return
}

// -----

func.func @empty_operator(%arg : i32) {
    // expected-error @+1 {{'emitc.apply' op applicable operator must not be empty}}
    %2 = emitc.apply ""(%arg) : (i32) -> !emitc.ptr<i32>
    return
}

// -----

func.func @illegal_operator(%arg : i32) {
    // expected-error @+1 {{'emitc.apply' op applicable operator is illegal}}
    %2 = emitc.apply "+"(%arg) : (i32) -> !emitc.ptr<i32>
    return
}

// -----

func.func @illegal_operand() {
    %1 = "emitc.constant"(){value = 42: i32} : () -> i32
    // expected-error @+1 {{'emitc.apply' op cannot apply to constant}}
    %2 = emitc.apply "&"(%1) : (i32) -> !emitc.ptr<i32>
    return
}

// -----

func.func @var_attribute_return_type_1() {
    // expected-error @+1 {{'emitc.variable' op requires attribute to either be an #emitc.opaque attribute or it's type ('i64') to match the op's result type ('i32')}}
    %c0 = "emitc.variable"(){value = 42: i64} : () -> i32
    return
}

// -----

func.func @var_attribute_return_type_2() {
    // expected-error @+1 {{'emitc.variable' op attribute 'value' failed to satisfy constraint: An opaque attribute or TypedAttr instance}}
    %c0 = "emitc.variable"(){value = unit} : () -> i32
    return
}

// -----

func.func @cast_tensor(%arg : tensor<f32>) {
    // expected-error @+1 {{'emitc.cast' op operand type 'tensor<f32>' and result type 'tensor<f32>' are cast incompatible}}
    %1 = emitc.cast %arg: tensor<f32> to tensor<f32>
    return
}

// -----

func.func @add_two_pointers(%arg0: !emitc.ptr<f32>, %arg1: !emitc.ptr<f32>) {
    // expected-error @+1 {{'emitc.add' op requires that at most one operand is a pointer}}
    %1 = "emitc.add" (%arg0, %arg1) : (!emitc.ptr<f32>, !emitc.ptr<f32>) -> !emitc.ptr<f32>
    return
}

// -----

func.func @add_pointer_float(%arg0: !emitc.ptr<f32>, %arg1: f32) {
    // expected-error @+1 {{'emitc.add' op requires that one operand is an integer or of opaque type if the other is a pointer}}
    %1 = "emitc.add" (%arg0, %arg1) : (!emitc.ptr<f32>, f32) -> !emitc.ptr<f32>
    return
}

// -----

func.func @add_float_pointer(%arg0: f32, %arg1: !emitc.ptr<f32>) {
    // expected-error @+1 {{'emitc.add' op requires that one operand is an integer or of opaque type if the other is a pointer}}
    %1 = "emitc.add" (%arg0, %arg1) : (f32, !emitc.ptr<f32>) -> !emitc.ptr<f32>
    return
}

// -----

func.func @div_tensor(%arg0: tensor<i32>, %arg1: tensor<i32>) {
    // expected-error @+1 {{'emitc.div' op operand #0 must be floating-point or integer or index or EmitC opaque type, but got 'tensor<i32>'}}
    %1 = "emitc.div" (%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return
}

// -----

func.func @mul_tensor(%arg0: tensor<i32>, %arg1: tensor<i32>) {
    // expected-error @+1 {{'emitc.mul' op operand #0 must be floating-point or integer or index or EmitC opaque type, but got 'tensor<i32>'}}
    %1 = "emitc.mul" (%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return
}

// -----

func.func @rem_tensor(%arg0: tensor<i32>, %arg1: tensor<i32>) {
    // expected-error @+1 {{'emitc.rem' op operand #0 must be integer or index or EmitC opaque type, but got 'tensor<i32>'}}
    %1 = "emitc.rem" (%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return
}

// -----

func.func @rem_float(%arg0: f32, %arg1: f32) {
    // expected-error @+1 {{'emitc.rem' op operand #0 must be integer or index or EmitC opaque type, but got 'f32'}}
    %1 = "emitc.rem" (%arg0, %arg1) : (f32, f32) -> f32
    return
}

// -----

func.func @sub_int_pointer(%arg0: i32, %arg1: !emitc.ptr<f32>) {
    // expected-error @+1 {{'emitc.sub' op rhs can only be a pointer if lhs is a pointer}}
    %1 = "emitc.sub" (%arg0, %arg1) : (i32, !emitc.ptr<f32>) -> !emitc.ptr<f32>
    return
}


// -----

func.func @sub_pointer_float(%arg0: !emitc.ptr<f32>, %arg1: f32) {
    // expected-error @+1 {{'emitc.sub' op requires that rhs is an integer, pointer or of opaque type if lhs is a pointer}}
    %1 = "emitc.sub" (%arg0, %arg1) : (!emitc.ptr<f32>, f32) -> !emitc.ptr<f32>
    return
}

// -----

func.func @sub_pointer_pointer(%arg0: !emitc.ptr<f32>, %arg1: !emitc.ptr<f32>) {
    // expected-error @+1 {{'emitc.sub' op requires that the result is an integer or of opaque type if lhs and rhs are pointers}}
    %1 = "emitc.sub" (%arg0, %arg1) : (!emitc.ptr<f32>, !emitc.ptr<f32>) -> !emitc.ptr<f32>
    return
}

// -----

func.func @test_misplaced_yield() {
  // expected-error @+1 {{'emitc.yield' op expects parent op to be one of 'emitc.expression, emitc.if, emitc.for'}}
  emitc.yield
  return
}

// -----

func.func @test_assign_to_non_variable(%arg1: f32, %arg2: f32) {
  // expected-error @+1 {{'emitc.assign' op requires first operand (<block argument> of type 'f32' at index: 1) to be a Variable}}
  emitc.assign %arg1 : f32 to %arg2 : f32
  return
}

// -----

func.func @test_assign_type_mismatch(%arg1: f32) {
  %v = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> i32
  // expected-error @+1 {{'emitc.assign' op requires value's type ('f32') to match variable's type ('i32')}}
  emitc.assign %arg1 : f32 to %v : i32
  return
}

// -----

func.func @test_expression_no_yield() -> i32 {
  // expected-error @+1 {{'emitc.expression' op must yield a value at termination}}
  %r = emitc.expression : i32 {
    %c7 = "emitc.constant"(){value = 7 : i32} : () -> i32
  }
  return %r : i32
}

// -----

func.func @test_expression_illegal_op(%arg0 : i1) -> i32 {
  // expected-error @+1 {{'emitc.expression' op contains an unsupported operation}}
  %r = emitc.expression : i32 {
    %x = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> i32
    emitc.yield %x : i32
  }
  return %r : i32
}

// -----

func.func @test_expression_no_use(%arg0: i32, %arg1: i32) -> i32 {
  // expected-error @+1 {{'emitc.expression' op requires exactly one use for each operation}}
  %r = emitc.expression : i32 {
    %a = emitc.add %arg0, %arg1 : (i32, i32) -> i32
    %b = emitc.rem %arg0, %arg1 : (i32, i32) -> i32
    emitc.yield %a : i32
  }
  return %r : i32
}

// -----

func.func @test_expression_multiple_uses(%arg0: i32, %arg1: i32) -> i32 {
  // expected-error @+1 {{'emitc.expression' op requires exactly one use for each operation}}
  %r = emitc.expression : i32 {
    %a = emitc.rem %arg0, %arg1 : (i32, i32) -> i32
    %b = emitc.add %a, %arg0 : (i32, i32) -> i32
    %c = emitc.mul %arg1, %a : (i32, i32) -> i32
    emitc.yield %a : i32
  }
  return %r : i32
}

// -----

func.func @test_expression_multiple_results(%arg0: i32) -> i32 {
  // expected-error @+1 {{'emitc.expression' op requires exactly one result for each operation}}
  %r = emitc.expression : i32 {
    %a:2 = emitc.call_opaque "bar" (%arg0) : (i32) -> (i32, i32)
    emitc.yield %a : i32
  }
  return %r : i32
}

// -----

// expected-error @+1 {{'emitc.func' op requires zero or exactly one result, but has 2}}
emitc.func @multiple_results(%0: i32) -> (i32, i32) {
  emitc.return %0 : i32
}

// -----

emitc.func @resulterror() -> i32 {
^bb42:
  emitc.return    // expected-error {{'emitc.return' op has 0 operands, but enclosing function (@resulterror) returns 1}}
}

// -----

emitc.func @return_type_mismatch() -> i32 {
  %0 = emitc.call_opaque "foo()"(): () -> f32
  emitc.return %0 : f32  // expected-error {{type of the return operand ('f32') doesn't match function result type ('i32') in function @return_type_mismatch}}
}

// -----

func.func @return_inside_func.func(%0: i32) -> (i32) {
  // expected-error@+1 {{'emitc.return' op expects parent op 'emitc.func'}}
  emitc.return %0 : i32
}
// -----

// expected-error@+1 {{expected non-function type}}
emitc.func @func_variadic(...)

// -----

// expected-error@+1 {{'emitc.func' op does not support empty function bodies}}
emitc.func private @empty()
