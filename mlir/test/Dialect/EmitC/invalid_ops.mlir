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

func.func @array_result() {
    // expected-error @+1 {{'emitc.call_opaque' op cannot return array type}}
    emitc.call_opaque "array_result"() : () -> !emitc.array<4xi32>
    return
}

// -----

func.func @empty_operator() {
    %0 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
    // expected-error @+1 {{'emitc.apply' op applicable operator must not be empty}}
    %1 = emitc.apply ""(%0) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
    return
}

// -----

func.func @illegal_operator() {
    %0 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
    // expected-error @+1 {{'emitc.apply' op applicable operator is illegal}}
    %1 = emitc.apply "+"(%0) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
    return
}

// -----

func.func @var_attribute_return_type_1() {
    // expected-error @+1 {{'emitc.variable' op requires attribute to either be an #emitc.opaque attribute or it's type ('i64') to match the op's result type ('i32')}}
    %c0 = "emitc.variable"(){value = 42: i64} : () -> !emitc.lvalue<i32>
    return
}

// -----

func.func @var_attribute_return_type_2() {
    // expected-error @+1 {{'emitc.variable' op attribute 'value' failed to satisfy constraint: An opaque attribute or TypedAttr instance}}
    %c0 = "emitc.variable"(){value = unit} : () -> !emitc.lvalue<i32>
    return
}

// -----

func.func @cast_tensor(%arg : tensor<f32>) {
    // expected-error @+1 {{'emitc.cast' op operand type 'tensor<f32>' and result type 'tensor<f32>' are cast incompatible}}
    %1 = emitc.cast %arg: tensor<f32> to tensor<f32>
    return
}

// -----

func.func @cast_array(%arg : !emitc.array<4xf32>) {
    // expected-error @+1 {{'emitc.cast' op operand type '!emitc.array<4xf32>' and result type '!emitc.array<4xf32>' are cast incompatible}}
    %1 = emitc.cast %arg: !emitc.array<4xf32> to !emitc.array<4xf32>
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
    // expected-error @+1 {{'emitc.div' op operand #0 must be floating-point type supported by EmitC or integer, index or opaque type supported by EmitC, but got 'tensor<i32>'}}
    %1 = "emitc.div" (%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return
}

// -----

func.func @mul_tensor(%arg0: tensor<i32>, %arg1: tensor<i32>) {
    // expected-error @+1 {{'emitc.mul' op operand #0 must be floating-point type supported by EmitC or integer, index or opaque type supported by EmitC, but got 'tensor<i32>'}}
    %1 = "emitc.mul" (%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return
}

// -----

func.func @rem_tensor(%arg0: tensor<i32>, %arg1: tensor<i32>) {
    // expected-error @+1 {{'emitc.rem' op operand #0 must be integer, index or opaque type supported by EmitC, but got 'tensor<i32>'}}
    %1 = "emitc.rem" (%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return
}

// -----

func.func @rem_float(%arg0: f32, %arg1: f32) {
    // expected-error @+1 {{'emitc.rem' op operand #0 must be integer, index or opaque type supported by EmitC, but got 'f32'}}
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
    // expected-error @+1 {{'emitc.sub' op requires that the result is an integer, ptrdiff_t or of opaque type if lhs and rhs are pointers}}
    %1 = "emitc.sub" (%arg0, %arg1) : (!emitc.ptr<f32>, !emitc.ptr<f32>) -> !emitc.ptr<f32>
    return
}

// -----

func.func @test_misplaced_yield() {
  // expected-error @+1 {{'emitc.yield' op expects parent op to be one of 'emitc.expression, emitc.if, emitc.for, emitc.switch'}}
  emitc.yield
  return
}

// -----

func.func @test_assign_to_block_argument(%arg0: f32) {
  %0 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<f32>
  cf.br ^bb1(%0 : !emitc.lvalue<f32>)
^bb1(%a : !emitc.lvalue<f32>):
  // expected-error @+1 {{'emitc.assign' op cannot assign to block argument}}
  emitc.assign %arg0 : f32 to %a : !emitc.lvalue<f32>
  func.return
}

// -----

func.func @test_assign_type_mismatch(%arg1: f32) {
  %v = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
  // expected-error @+1 {{'emitc.assign' op requires value's type ('f32') to match variable's type ('i32')}}
  emitc.assign %arg1 : f32 to %v : !emitc.lvalue<i32>
  return
}

// -----

func.func @test_assign_to_array(%arg1: !emitc.array<4xi32>) {
  %v = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<4xi32>
  // expected-error @+1 {{invalid kind of Type specified}}
  emitc.assign %arg1 : !emitc.array<4xi32> to %v : !emitc.array<4xi32>
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
    %x = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
    %y = emitc.load %x : <i32>
    emitc.yield %y : i32
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

// expected-error@+1 {{'emitc.func' op cannot have lvalue type as argument}}
emitc.func @argument_type_lvalue(%arg : !emitc.lvalue<i32>) {
  emitc.return
}

// -----

// expected-error@+1 {{'emitc.func' op cannot return array type}}
emitc.func @return_type_array(%arg : !emitc.array<4xi32>) -> !emitc.array<4xi32> {
  emitc.return %arg : !emitc.array<4xi32>
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

// expected-error@+1 {{'emitc.declare_func' op 'bar' does not reference a valid function}}
emitc.declare_func @bar

// -----

// expected-error@+1 {{'emitc.declare_func' op requires attribute 'sym_name'}}
"emitc.declare_func"()  : () -> ()

// -----

func.func @logical_and_resulterror(%arg0: i32, %arg1: i32) {
  // expected-error @+1 {{'emitc.logical_and' op result #0 must be 1-bit signless integer, but got 'i32'}}
  %0 = "emitc.logical_and"(%arg0, %arg1) : (i32, i32) -> i32
  return
}

// -----

func.func @logical_not_resulterror(%arg0: i32) {
  // expected-error @+1 {{'emitc.logical_not' op result #0 must be 1-bit signless integer, but got 'i32'}}
  %0 = "emitc.logical_not"(%arg0) : (i32) -> i32
  return
}

// -----

func.func @logical_or_resulterror(%arg0: i32, %arg1: i32) {
  // expected-error @+1 {{'emitc.logical_or' op result #0 must be 1-bit signless integer, but got 'i32'}}
  %0 = "emitc.logical_or"(%arg0, %arg1) : (i32, i32) -> i32
  return
}

// -----

func.func @test_subscript_array_indices_mismatch(%arg0: !emitc.array<4x8xf32>, %arg1: index) {
  // expected-error @+1 {{'emitc.subscript' op on array operand requires number of indices (1) to match the rank of the array type (2)}}
  %0 = emitc.subscript %arg0[%arg1] : (!emitc.array<4x8xf32>, index) -> !emitc.lvalue<f32>
  return
}

// -----

func.func @test_subscript_array_index_type_mismatch(%arg0: !emitc.array<4x8xf32>, %arg1: index, %arg2: f32) {
  // expected-error @+1 {{'emitc.subscript' op on array operand requires index operand 1 to be integer-like, but got 'f32'}}
  %0 = emitc.subscript %arg0[%arg1, %arg2] : (!emitc.array<4x8xf32>, index, f32) -> !emitc.lvalue<f32>
  return
}

// -----

func.func @test_subscript_array_type_mismatch(%arg0: !emitc.array<4x8xf32>, %arg1: index, %arg2: index) {
  // expected-error @+1 {{'emitc.subscript' op on array operand requires element type ('f32') and result type ('i32') to match}}
  %0 = emitc.subscript %arg0[%arg1, %arg2] : (!emitc.array<4x8xf32>, index, index) -> !emitc.lvalue<i32>
  return
}

// -----

func.func @test_subscript_ptr_indices_mismatch(%arg0: !emitc.ptr<f32>, %arg1: index) {
  // expected-error @+1 {{'emitc.subscript' op on pointer operand requires one index operand, but got 2}}
  %0 = emitc.subscript %arg0[%arg1, %arg1] : (!emitc.ptr<f32>, index, index) -> !emitc.lvalue<f32>
  return
}

// -----

func.func @test_subscript_ptr_index_type_mismatch(%arg0: !emitc.ptr<f32>, %arg1: f64) {
  // expected-error @+1 {{'emitc.subscript' op on pointer operand requires index operand to be integer-like, but got 'f64'}}
  %0 = emitc.subscript %arg0[%arg1] : (!emitc.ptr<f32>, f64) -> !emitc.lvalue<f32>
  return
}

// -----

func.func @test_subscript_ptr_type_mismatch(%arg0: !emitc.ptr<f32>, %arg1: index) {
  // expected-error @+1 {{'emitc.subscript' op on pointer operand requires pointee type ('f32') and result type ('f64') to match}}
  %0 = emitc.subscript %arg0[%arg1] : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f64>
  return
}

// -----

// expected-error @+1 {{'emitc.global' op cannot have both static and extern specifiers}}
emitc.global extern static @uninit : i32

// -----

emitc.global @myglobal_array : !emitc.array<2xf32>

func.func @use_global() {
  // expected-error @+1 {{'emitc.get_global' op on array type expects result type '!emitc.array<3xf32>' to match type '!emitc.array<2xf32>' of the global @myglobal_array}}
  %0 = emitc.get_global @myglobal_array : !emitc.array<3xf32>
  return
}

// -----

emitc.global @myglobal_scalar : f32

func.func @use_global() {
  // expected-error @+1 {{'emitc.get_global' op on non-array type expects result inner type 'i32' to match type 'f32' of the global @myglobal_scalar}}
  %0 = emitc.get_global @myglobal_scalar : !emitc.lvalue<i32>
  return
}

// -----

func.func @member(%arg0: !emitc.lvalue<i32>) {
  // expected-error @+1 {{'emitc.member' op operand #0 must be emitc.lvalue of EmitC opaque type values, but got '!emitc.lvalue<i32>'}}
  %0 = "emitc.member" (%arg0) {member = "a"} : (!emitc.lvalue<i32>) -> !emitc.lvalue<i32>
  return
}

// -----

func.func @member_of_ptr(%arg0: !emitc.lvalue<i32>) {
  // expected-error @+1 {{'emitc.member_of_ptr' op operand #0 must be emitc.lvalue of EmitC opaque type or EmitC pointer type values, but got '!emitc.lvalue<i32>'}}
  %0 = "emitc.member_of_ptr" (%arg0) {member = "a"} : (!emitc.lvalue<i32>) -> !emitc.lvalue<i32>
  return
}

// -----

func.func @emitc_switch() {
  %0 = "emitc.constant"(){value = 1 : i16} : () -> i16

  // expected-error@+1 {{'emitc.switch' op expected region to end with emitc.yield, but got emitc.call_opaque}}
  emitc.switch %0 : i16
  case 2 {
    %1 = emitc.call_opaque "func_b" () : () -> i32
  }
  case 5 {
    %2 = emitc.call_opaque "func_a" () : () -> i32
    emitc.yield
  }
  default {
    %3 = "emitc.constant"(){value = 42.0 : f32} : () -> f32
    emitc.call_opaque "func2" (%3) : (f32) -> ()
    emitc.yield
  }
  return
}

// -----

func.func @emitc_switch() {
  %0 = "emitc.constant"(){value = 1 : i32} : () -> i32

  emitc.switch %0 : i32
  case 2 {
    %1 = emitc.call_opaque "func_b" () : () -> i32
    emitc.yield
  }
  // expected-error@+1 {{custom op 'emitc.switch' expected integer value}}
  case {
    %2 = emitc.call_opaque "func_a" () : () -> i32
    emitc.yield
  }
  default {
    %3 = "emitc.constant"(){value = 42.0 : f32} : () -> f32
    emitc.call_opaque "func2" (%3) : (f32) -> ()
    emitc.yield
  }
  return
}

// -----

func.func @emitc_switch() {
  %0 = "emitc.constant"(){value = 1 : i8} : () -> i8

  emitc.switch %0 : i8
  case 2 {
    %1 = emitc.call_opaque "func_b" () : () -> i32
    emitc.yield
  }
  case 3 {
    %2 = emitc.call_opaque "func_a" () : () -> i32
    emitc.yield
  }
  // expected-error@+1 {{custom op 'emitc.switch' expected 'default'}}
  return
}

// -----

func.func @emitc_switch() {
  %0 = "emitc.constant"(){value = 1 : i64} : () -> i64

  // expected-error@+1 {{'emitc.switch' op has duplicate case value: 2}}
  emitc.switch %0 : i64
  case 2 {
    %1 = emitc.call_opaque "func_b" () : () -> i32
    emitc.yield
  }
  case 2 {
    %2 = emitc.call_opaque "func_a" () : () -> i32
    emitc.yield
  }
  default {
    %3 = "emitc.constant"(){value = 42.0 : f32} : () -> f32
    emitc.call_opaque "func2" (%3) : (f32) -> ()
    emitc.yield
  }
  return
}
