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

func.func @c() {
  %1 = "emitc.constant"(){value = 42 : i32} : () -> i32
  return
}

func.func @a(%arg0: i32, %arg1: i32) {
  %1 = "emitc.apply"(%arg0) {applicableOperator = "&"} : (i32) -> !emitc.ptr<i32>
  %2 = emitc.apply "&"(%arg1) : (i32) -> !emitc.ptr<i32>
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
  %v = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
  emitc.assign %arg1 : f32 to %v : f32
  return
}

func.func @test_expression(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: f32, %arg4: f32) -> i32 {
  %c7 = "emitc.constant"() {value = 7 : i32} : () -> i32
  %q = emitc.expression : i32 {
    %a = emitc.rem %arg1, %c7 : (i32, i32) -> i32
    emitc.yield %a : i32
  }
  %r = emitc.expression noinline : i32 {
    %a = emitc.add %arg0, %arg1 : (i32, i32) -> i32
    %b = emitc.call_opaque "bar" (%a, %arg2, %q) : (i32, i32, i32) -> (i32)
    %c = emitc.mul %arg3, %arg4 : (f32, f32) -> f32
    %d = emitc.cast %c : f32 to i32
    %e = emitc.sub %b, %d : (i32, i32) -> i32
    emitc.yield %e : i32
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

emitc.verbatim "#ifdef __cplusplus"
emitc.verbatim "extern \"C\" {"
emitc.verbatim "#endif  // __cplusplus"

emitc.verbatim "#ifdef __cplusplus"
emitc.verbatim "}  // extern \"C\""
emitc.verbatim "#endif  // __cplusplus"

emitc.verbatim "typedef int32_t i32;"
emitc.verbatim "typedef float f32;"
