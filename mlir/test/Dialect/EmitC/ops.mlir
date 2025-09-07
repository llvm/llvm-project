// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s -canonicalize | FileCheck %s

// CHECK: emitc.include <"test.h">
// CHECK: emitc.include "test.h"
emitc.include <"test.h">
emitc.include "test.h"

// CHECK-LABEL: func @f(%{{.*}}: i32, %{{.*}}: !emitc.opaque<"int32_t">) {
func.func @f(%arg0: i32, %f: !emitc.opaque<"int32_t">) {
  %1 = "emitc.call_opaque"() {callee = "blah"} : () -> i64
  emitc.call_opaque "foo" (%1) {args = [
    0 : index, dense<[0, 1]> : tensor<2xi32>, 0 : index
  ]} : (i64) -> ()
  return
}

emitc.declare_func @func

emitc.func @func(%arg0 : i32) {
  emitc.call_opaque "foo"(%arg0) : (i32) -> ()
  emitc.return
}

emitc.func @return_i32() -> i32 attributes {specifiers = ["static","inline"]} {
  %0 = emitc.call_opaque "foo"(): () -> i32
  emitc.return %0 : i32
}

emitc.func @call() -> i32 {
  %0 = emitc.call @return_i32() : () -> (i32)
  emitc.return %0 : i32
}

emitc.func private @extern(i32) attributes {specifiers = ["extern"]}

func.func @cast(%arg0: i32) {
  %1 = emitc.cast %arg0: i32 to f32
  return
}

func.func @cast_array_to_pointer(%arg0: !emitc.array<3xi32>) {
  %1 = emitc.cast %arg0: !emitc.array<3xi32> to !emitc.ptr<i32>
  return
}

func.func @c() {
  %1 = "emitc.constant"(){value = 42 : i32} : () -> i32
  %2 = "emitc.constant"(){value = 42 : index} : () -> !emitc.size_t
  %3 = "emitc.constant"(){value = 42 : index} : () -> !emitc.ssize_t
  %4 = "emitc.constant"(){value = 42 : index} : () -> !emitc.ptrdiff_t
  return
}

func.func @a() {
  %0 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
  %1 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
  %2 = "emitc.apply"(%0) {applicableOperator = "&"} : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
  %3 = emitc.apply "&"(%1) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
  return
}

func.func @add_int(%arg0: i32, %arg1: i32) {
  %1 = "emitc.add" (%arg0, %arg1) : (i32, i32) -> i32
  return
}

func.func @add_pointer(%arg0: !emitc.ptr<f32>, %arg1: i32, %arg2: !emitc.opaque<"unsigned int">) {
  %1 = "emitc.add" (%arg0, %arg1) : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
  %2 = "emitc.add" (%arg0, %arg2) : (!emitc.ptr<f32>, !emitc.opaque<"unsigned int">) -> !emitc.ptr<f32>
  return
}

func.func @bitwise(%arg0: i32, %arg1: i32) -> () {
  %0 = emitc.bitwise_and %arg0, %arg1 : (i32, i32) -> i32
  %1 = emitc.bitwise_left_shift %arg0, %arg1 : (i32, i32) -> i32
  %2 = emitc.bitwise_not %arg0 : (i32) -> i32
  %3 = emitc.bitwise_or %arg0, %arg1 : (i32, i32) -> i32
  %4 = emitc.bitwise_right_shift %arg0, %arg1 : (i32, i32) -> i32
  %5 = emitc.bitwise_xor %arg0, %arg1 : (i32, i32) -> i32
  return
}

func.func @cond(%cond: i1, %arg0: i32, %arg1: i32) -> () {
  %0 = emitc.conditional %cond, %arg0, %arg1 : i32
  return
}

func.func @div_int(%arg0: i32, %arg1: i32) {
  %1 = "emitc.div" (%arg0, %arg1) : (i32, i32) -> i32
  return
}

func.func @div_float(%arg0: f32, %arg1: f32) {
  %1 = "emitc.div" (%arg0, %arg1) : (f32, f32) -> f32
  return
}

func.func @mul_int(%arg0: i32, %arg1: i32) {
  %1 = "emitc.mul" (%arg0, %arg1) : (i32, i32) -> i32
  return
}

func.func @mul_float(%arg0: f32, %arg1: f32) {
  %1 = "emitc.mul" (%arg0, %arg1) : (f32, f32) -> f32
  return
}

func.func @rem(%arg0: i32, %arg1: i32) {
  %1 = "emitc.rem" (%arg0, %arg1) : (i32, i32) -> i32
  return
}

func.func @sub_int(%arg0: i32, %arg1: i32) {
  %1 = "emitc.sub" (%arg0, %arg1) : (i32, i32) -> i32
  return
}

func.func @sub_pointer(%arg0: !emitc.ptr<f32>, %arg1: i32, %arg2: !emitc.opaque<"unsigned int">, %arg3: !emitc.ptr<f32>) {
  %1 = "emitc.sub" (%arg0, %arg1) : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
  %2 = "emitc.sub" (%arg0, %arg2) : (!emitc.ptr<f32>, !emitc.opaque<"unsigned int">) -> !emitc.ptr<f32>
  %3 = "emitc.sub" (%arg0, %arg3) : (!emitc.ptr<f32>, !emitc.ptr<f32>) -> !emitc.opaque<"ptrdiff_t">
  %4 = "emitc.sub" (%arg0, %arg3) : (!emitc.ptr<f32>, !emitc.ptr<f32>) -> i32
  %5 = "emitc.sub" (%arg0, %arg3) : (!emitc.ptr<f32>, !emitc.ptr<f32>) -> !emitc.ptrdiff_t
  return
}

func.func @cmp(%arg0 : i32, %arg1 : f32, %arg2 : i64, %arg3 : f64, %arg4 : !emitc.opaque<"unsigned">, %arg5 : !emitc.opaque<"std::valarray<int>">, %arg6 : !emitc.opaque<"custom">) {
  %1 = "emitc.cmp" (%arg0, %arg0) {predicate = 0} : (i32, i32) -> i1
  %2 = emitc.cmp eq, %arg0, %arg0 : (i32, i32) -> i1
  %3 = "emitc.cmp" (%arg1, %arg1) {predicate = 1} : (f32, f32) -> i1
  %4 = emitc.cmp ne, %arg1, %arg1 : (f32, f32) -> i1
  %5 = "emitc.cmp" (%arg2, %arg2) {predicate = 2} : (i64, i64) -> i1
  %6 = emitc.cmp lt, %arg2, %arg2 : (i64, i64) -> i1
  %7 = "emitc.cmp" (%arg3, %arg3) {predicate = 3} : (f64, f64) -> i1
  %8 = emitc.cmp le, %arg3, %arg3 : (f64, f64) -> i1
  %9 = "emitc.cmp" (%arg4, %arg4) {predicate = 4} : (!emitc.opaque<"unsigned">, !emitc.opaque<"unsigned">) -> i1
  %10 = emitc.cmp gt, %arg4, %arg4 : (!emitc.opaque<"unsigned">, !emitc.opaque<"unsigned">) -> i1
  %11 = "emitc.cmp" (%arg5, %arg5) {predicate = 5} : (!emitc.opaque<"std::valarray<int>">, !emitc.opaque<"std::valarray<int>">) -> !emitc.opaque<"std::valarray<bool>">
  %12 = emitc.cmp ge, %arg5, %arg5 : (!emitc.opaque<"std::valarray<int>">, !emitc.opaque<"std::valarray<int>">) -> !emitc.opaque<"std::valarray<bool>">
  %13 = "emitc.cmp" (%arg6, %arg6) {predicate = 6} : (!emitc.opaque<"custom">, !emitc.opaque<"custom">) -> !emitc.opaque<"custom">
  %14 = emitc.cmp three_way, %arg6, %arg6 : (!emitc.opaque<"custom">, !emitc.opaque<"custom">) -> !emitc.opaque<"custom">
  return
}

func.func @logical(%arg0: i32, %arg1: i32) {
  %0 = emitc.logical_and %arg0, %arg1 : i32, i32
  %1 = emitc.logical_not %arg0 : i32
  %2 = emitc.logical_or %arg0, %arg1 : i32, i32
  return
}

func.func @unary(%arg0: i32) {
  %0 = emitc.unary_minus %arg0 : (i32) -> i32
  %1 = emitc.unary_plus %arg0 : (i32) -> i32
  return
}

func.func @test_if(%arg0: i1, %arg1: f32) {
  emitc.if %arg0 {
     %0 = emitc.call_opaque "func_const"(%arg1) : (f32) -> i32
  }
  return
}

func.func @test_if_explicit_yield(%arg0: i1, %arg1: f32) {
  emitc.if %arg0 {
     %0 = emitc.call_opaque "func_const"(%arg1) : (f32) -> i32
     emitc.yield
  }
  return
}

func.func @test_if_else(%arg0: i1, %arg1: f32) {
  emitc.if %arg0 {
    %0 = emitc.call_opaque "func_true"(%arg1) : (f32) -> i32
  } else {
    %0 = emitc.call_opaque "func_false"(%arg1) : (f32) -> i32
  }
  return
}

func.func @test_assign(%arg1: f32) {
  %v = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<f32>
  emitc.assign %arg1 : f32 to %v : !emitc.lvalue<f32>
  return
}

func.func @test_expression(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: f32, %arg4: f32) -> i32 {
  %c7 = "emitc.constant"() {value = 7 : i32} : () -> i32
  %q = emitc.expression %arg1, %c7 : (i32, i32) -> i32 {
    %a = emitc.rem %arg1, %c7 : (i32, i32) -> i32
    emitc.yield %a : i32
  }
  %r = emitc.expression %arg0, %arg1, %arg2, %arg3, %arg4, %q noinline : (i32, i32, i32, f32, f32, i32) -> i32 {
    %a = emitc.add %arg0, %arg1 : (i32, i32) -> i32
    %b = emitc.call_opaque "bar" (%a, %arg2, %q) : (i32, i32, i32) -> (i32)
    %c = emitc.mul %arg3, %arg4 : (f32, f32) -> f32
    %d = emitc.cast %c : f32 to i32
    %e = emitc.sub %b, %d : (i32, i32) -> i32
    emitc.yield %e : i32
  }
  return %r : i32
}

func.func @test_expression_multiple_uses(%arg0: i32, %arg1: i32) -> i32 {
  %r = emitc.expression %arg0, %arg1 : (i32, i32) -> i32 {
    %a = emitc.rem %arg0, %arg1 : (i32, i32) -> i32
    %b = emitc.add %a, %arg0 : (i32, i32) -> i32
    %c = emitc.mul %b, %a : (i32, i32) -> i32
    emitc.yield %c : i32
  }
  return %r : i32
}

func.func @test_for(%arg0 : index, %arg1 : index, %arg2 : index) {
  emitc.for %i0 = %arg0 to %arg1 step %arg2 {
    %0 = emitc.call_opaque "func_const"(%i0) : (index) -> i32
  }
  return
}

func.func @test_for_explicit_yield(%arg0 : index, %arg1 : index, %arg2 : index) {
  emitc.for %i0 = %arg0 to %arg1 step %arg2 {
    %0 = emitc.call_opaque "func_const"(%i0) : (index) -> i32
    emitc.yield
  }
  return
}

func.func @test_for_not_index_induction(%arg0 : i16, %arg1 : i16, %arg2 : i16) {
  emitc.for %i0 = %arg0 to %arg1 step %arg2 : i16 {
    %0 = emitc.call_opaque "func_const"(%i0) : (i16) -> i32
  }
  return
}

func.func @test_subscript(%arg0 : !emitc.array<2x3xf32>, %arg1 : !emitc.ptr<i32>, %arg2 : !emitc.opaque<"std::map<char, int>">, %idx0 : index, %idx1 : i32, %idx2 : !emitc.opaque<"char">) {
  %0 = emitc.subscript %arg0[%idx0, %idx1] : (!emitc.array<2x3xf32>, index, i32) -> !emitc.lvalue<f32>
  %1 = emitc.subscript %arg1[%idx0] : (!emitc.ptr<i32>, index) -> !emitc.lvalue<i32>
  %2 = emitc.subscript %arg2[%idx2] : (!emitc.opaque<"std::map<char, int>">, !emitc.opaque<"char">) -> !emitc.lvalue<!emitc.opaque<"int">>
  return
}

emitc.verbatim "#ifdef __cplusplus"
emitc.verbatim "extern \"C\" {"
emitc.verbatim "#endif  // __cplusplus"

emitc.verbatim "#ifdef __cplusplus"
emitc.verbatim "}  // extern \"C\""
emitc.verbatim "#endif  // __cplusplus"

emitc.verbatim "typedef int32_t i32;"
emitc.verbatim "typedef float f32;"

// The value is not interpreted as format string if there are no operands.
emitc.verbatim "{} {  }"

func.func @test_verbatim(%arg0 : !emitc.ptr<i32>, %arg1 : i32, %arg2: !emitc.array<3x!emitc.ptr<i32>>) {
  %a = "emitc.variable"() <{value = #emitc.opaque<"1">}> : () -> !emitc.lvalue<i32>

  // Check that the lvalue type can be used by verbatim.
  emitc.verbatim "++{};" args %a : !emitc.lvalue<i32>

  // Check that the array type can be used by verbatim.
  emitc.verbatim "*{}[0] = 1;" args %arg2 : !emitc.array<3x!emitc.ptr<i32>>

  emitc.verbatim "{} + {};" args %arg0, %arg1 : !emitc.ptr<i32>, i32

  // Check there is no ambiguity whether %b is the argument to the emitc.verbatim op.
  emitc.verbatim "b"
  %b = "emitc.constant"(){value = 42 : i32} : () -> i32

  return
}

emitc.global @uninit : i32
emitc.global @myglobal_int : i32 = 4
emitc.global extern @external_linkage : i32
emitc.global static @internal_linkage : i32
emitc.global @myglobal : !emitc.array<2xf32> = dense<4.000000e+00>
emitc.global const @myconstant : !emitc.array<2xi16> = dense<2>

func.func @use_global(%i: index) -> f32 {
  %0 = emitc.get_global @myglobal : !emitc.array<2xf32>
  %1 = emitc.subscript %0[%i] : (!emitc.array<2xf32>, index) -> !emitc.lvalue<f32>
  %2 = emitc.load %1 : <f32>
  return %2 : f32
}

func.func @assign_global(%arg0 : i32) {
  %0 = emitc.get_global @myglobal_int : !emitc.lvalue<i32>
  emitc.assign %arg0 : i32 to %0 : !emitc.lvalue<i32>
  return
}

func.func @member_access(%arg0: !emitc.lvalue<!emitc.opaque<"mystruct">>, %arg1: !emitc.lvalue<!emitc.opaque<"mystruct_ptr">>, %arg2: !emitc.lvalue<!emitc.ptr<!emitc.opaque<"mystruct">>>) {
  %0 = "emitc.member" (%arg0) {member = "a"} : (!emitc.lvalue<!emitc.opaque<"mystruct">>) -> !emitc.lvalue<i32>
  %1 = "emitc.member" (%arg0) {member = "b"} : (!emitc.lvalue<!emitc.opaque<"mystruct">>) -> !emitc.array<2xi32>
  %2 = "emitc.member_of_ptr" (%arg1) {member = "a"} : (!emitc.lvalue<!emitc.opaque<"mystruct_ptr">>) -> !emitc.lvalue<i32>
  %3 = "emitc.member_of_ptr" (%arg1) {member = "b"} : (!emitc.lvalue<!emitc.opaque<"mystruct_ptr">>) -> !emitc.array<2xi32>
  %4 = "emitc.member_of_ptr" (%arg2) {member = "a"} : (!emitc.lvalue<!emitc.ptr<!emitc.opaque<"mystruct">>>) -> !emitc.lvalue<i32>
  %5 = "emitc.member_of_ptr" (%arg2) {member = "b"} : (!emitc.lvalue<!emitc.ptr<!emitc.opaque<"mystruct">>>) -> !emitc.array<2xi32>
  return
}

func.func @switch() {
  %0 = "emitc.constant"(){value = 1 : index} : () -> !emitc.ptrdiff_t

  emitc.switch %0 : !emitc.ptrdiff_t
  case 1 {
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
  }

  return 
}

emitc.class final @finalClass {
  emitc.field @fieldName0 : !emitc.array<1xf32>
  emitc.field @fieldName1 : !emitc.array<1xf32>
  emitc.func @execute() {
    %0 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
    %1 = get_field @fieldName0 : !emitc.array<1xf32>
    %2 = get_field @fieldName1 : !emitc.array<1xf32>
    %3 = subscript %1[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
    return
  }
}
