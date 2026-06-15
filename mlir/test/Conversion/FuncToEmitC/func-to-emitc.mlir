// RUN: mlir-opt -split-input-file -convert-func-to-emitc="lower-to-cpp=false" %s | FileCheck %s
// RUN: mlir-opt -split-input-file -convert-to-emitc="filter-dialects=func lower-to-cpp=false" %s | FileCheck %s

// CHECK-LABEL: emitc.func @foo()
// CHECK-NEXT: return
func.func @foo() {
  return
}

// -----

// CHECK-LABEL: emitc.func private @foo() attributes {specifiers = ["static"]}
// CHECK-NEXT: return
func.func private @foo() {
  return
}

// -----

// CHECK-LABEL: emitc.func @foo(%arg0: i32)
func.func @foo(%arg0: i32) {
  emitc.call_opaque "bar"(%arg0) : (i32) -> ()
  return
}

// -----

// CHECK-LABEL: emitc.func @foo(%arg0: i32) -> i32
// CHECK-NEXT: return %arg0 : i32
func.func @foo(%arg0: i32) -> i32 {
  return %arg0 : i32
}

// -----

// CHECK-LABEL: emitc.func @foo(%arg0: i32, %arg1: i32) -> i32
func.func @foo(%arg0: i32, %arg1: i32) -> i32 {
  %0 = "emitc.add" (%arg0, %arg1) : (i32, i32) -> i32
  return %0 : i32
}

// -----

// CHECK-LABEL: emitc.func private @return_i32(%arg0: i32) -> i32 attributes {specifiers = ["static"]}
// CHECK-NEXT: return %arg0 : i32
func.func private @return_i32(%arg0: i32) -> i32 {
  return %arg0 : i32
}

// CHECK-LABEL: emitc.func @call(%arg0: i32) -> i32
// CHECK-NEXT: %0 = call @return_i32(%arg0) : (i32) -> i32
// CHECK-NEXT: return %0 : i32
func.func @call(%arg0: i32) -> i32 {
  %0 = call @return_i32(%arg0) : (i32) -> (i32)
  return %0 : i32
}

// -----

// CHECK-LABEL: emitc.func private @return_i32(i32) -> i32 attributes {specifiers = ["extern"]}
func.func private @return_i32(%arg0: i32) -> i32

// -----

// CHECK-LABEL: emitc.func private @return_void() attributes {specifiers = ["static"]}
// CHECK-NEXT: return
func.func private @return_void() {
  return
}

// CHECK-LABEL: emitc.func @call()
// CHECK-NEXT: call @return_void() : () -> ()
// CHECK-NEXT: return
func.func @call() {
  call @return_void() : () -> ()
  return
}

// -----

// Multi-result function: check that an emitc.class struct is created and the
// function returns the packed struct.
// CHECK-LABEL:   emitc.class struct @return_i32_i32 {
// CHECK:           emitc.field @field0 : i32
// CHECK:           emitc.field @field1 : i32
// CHECK:         }
// CHECK-LABEL:   emitc.func @return_two(
// CHECK-SAME:      %[[ARG0:.*]]: i32,
// CHECK-SAME:      %[[ARG1:.*]]: i32) -> !emitc.opaque<"struct return_i32_i32"> {
// CHECK:           %[[VAL_0:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"struct return_i32_i32">>
// CHECK:           %[[VAL_1:.*]] = "emitc.member"(%[[VAL_0]]) <{member = "field0"}> : (!emitc.lvalue<!emitc.opaque<"struct return_i32_i32">>) -> !emitc.lvalue<i32>
// CHECK:           assign %[[ARG0]] : i32 to %[[VAL_1]] : <i32>
// CHECK:           %[[VAL_2:.*]] = "emitc.member"(%[[VAL_0]]) <{member = "field1"}> : (!emitc.lvalue<!emitc.opaque<"struct return_i32_i32">>) -> !emitc.lvalue<i32>
// CHECK:           assign %[[ARG1]] : i32 to %[[VAL_2]] : <i32>
// CHECK:           %[[VAL_3:.*]] = load %[[VAL_0]] : <!emitc.opaque<"struct return_i32_i32">>
// CHECK:           return %[[VAL_3]] : !emitc.opaque<"struct return_i32_i32">
// CHECK:         }
func.func @return_two(%arg0: i32, %arg1: i32) -> (i32, i32) {
  return %arg0, %arg1 : i32, i32
}

// -----

// Call to a multi-result function: check that the call returns the struct and
// that only the field actually used is extracted.
// CHECK-LABEL:   emitc.class struct @return_i32_i32 {
// CHECK:           emitc.field @field0 : i32
// CHECK:           emitc.field @field1 : i32
// CHECK:         }
// CHECK-LABEL:   emitc.func @return_two(
// CHECK-SAME:      %[[ARG0:.*]]: i32,
// CHECK-SAME:      %[[ARG1:.*]]: i32) -> !emitc.opaque<"struct return_i32_i32"> {
// CHECK-NEXT:      %[[VAL_0:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"struct return_i32_i32">>
// CHECK-NEXT:      %[[VAL_1:.*]] = "emitc.member"(%[[VAL_0]]) <{member = "field0"}> : (!emitc.lvalue<!emitc.opaque<"struct return_i32_i32">>) -> !emitc.lvalue<i32>
// CHECK-NEXT:      assign %[[ARG0]] : i32 to %[[VAL_1]] : <i32>
// CHECK-NEXT:      %[[VAL_2:.*]] = "emitc.member"(%[[VAL_0]]) <{member = "field1"}> : (!emitc.lvalue<!emitc.opaque<"struct return_i32_i32">>) -> !emitc.lvalue<i32>
// CHECK-NEXT:      assign %[[ARG1]] : i32 to %[[VAL_2]] : <i32>
// CHECK-NEXT:      %[[VAL_3:.*]] = load %[[VAL_0]] : <!emitc.opaque<"struct return_i32_i32">>
// CHECK-NEXT:      return %[[VAL_3]] : !emitc.opaque<"struct return_i32_i32">
// CHECK-NEXT:    }
// CHECK-LABEL:   emitc.func @caller(
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> i32 {
// CHECK-NEXT:      %[[VAL_0:.*]] = call @return_two(%[[ARG0]], %[[ARG0]]) : (i32, i32) -> !emitc.opaque<"struct return_i32_i32">
// CHECK-NEXT:      %[[VAL_1:.*]] = "emitc.member"(%[[VAL_0]]) <{member = "field1"}> : (!emitc.opaque<"struct return_i32_i32">) -> i32
// CHECK-NEXT:      return %[[VAL_1]] : i32
// CHECK-NEXT:    }
func.func @return_two(%arg0: i32, %arg1: i32) -> (i32, i32) {
  return %arg0, %arg1 : i32, i32
}
func.func @caller(%arg0: i32) -> i32 {
  %0, %1 = call @return_two(%arg0, %arg0) : (i32, i32) -> (i32, i32)
  return %1 : i32
}

// -----

// Two functions returning the same type tuple share one emitc.class.
// CHECK-LABEL:   emitc.class struct @return_i32_i32 {
// CHECK:           emitc.field @field0 : i32
// CHECK:           emitc.field @field1 : i32
// CHECK:         }
// CHECK-LABEL:   emitc.func @first(
// CHECK-SAME:                      %[[ARG0:.*]]: i32) -> !emitc.opaque<"struct return_i32_i32"> {
// CHECK-NEXT:      %[[VAL_0:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"struct return_i32_i32">>
// CHECK-NEXT:      %[[VAL_1:.*]] = "emitc.member"(%[[VAL_0]]) <{member = "field0"}> : (!emitc.lvalue<!emitc.opaque<"struct return_i32_i32">>) -> !emitc.lvalue<i32>
// CHECK-NEXT:      assign %[[ARG0]] : i32 to %[[VAL_1]] : <i32>
// CHECK-NEXT:      %[[VAL_2:.*]] = "emitc.member"(%[[VAL_0]]) <{member = "field1"}> : (!emitc.lvalue<!emitc.opaque<"struct return_i32_i32">>) -> !emitc.lvalue<i32>
// CHECK-NEXT:      assign %[[ARG0]] : i32 to %[[VAL_2]] : <i32>
// CHECK-NEXT:      %[[VAL_3:.*]] = load %[[VAL_0]] : <!emitc.opaque<"struct return_i32_i32">>
// CHECK-NEXT:      return %[[VAL_3]] : !emitc.opaque<"struct return_i32_i32">
// CHECK-NEXT:    }
// CHECK-LABEL:   emitc.func @second(
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> !emitc.opaque<"struct return_i32_i32"> {
// CHECK-NEXT:      %[[VAL_0:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"struct return_i32_i32">>
// CHECK-NEXT:      %[[VAL_1:.*]] = "emitc.member"(%[[VAL_0]]) <{member = "field0"}> : (!emitc.lvalue<!emitc.opaque<"struct return_i32_i32">>) -> !emitc.lvalue<i32>
// CHECK-NEXT:      assign %[[ARG0]] : i32 to %[[VAL_1]] : <i32>
// CHECK-NEXT:      %[[VAL_2:.*]] = "emitc.member"(%[[VAL_0]]) <{member = "field1"}> : (!emitc.lvalue<!emitc.opaque<"struct return_i32_i32">>) -> !emitc.lvalue<i32>
// CHECK-NEXT:      assign %[[ARG0]] : i32 to %[[VAL_2]] : <i32>
// CHECK-NEXT:      %[[VAL_3:.*]] = load %[[VAL_0]] : <!emitc.opaque<"struct return_i32_i32">>
// CHECK-NEXT:      return %[[VAL_3]] : !emitc.opaque<"struct return_i32_i32">
// CHECK-NEXT:    }
// CHECK-NOT:     emitc.class
func.func @first(%arg0: i32) -> (i32, i32) {
  return %arg0, %arg0 : i32, i32
}
func.func @second(%arg0: i32) -> (i32, i32) {
  return %arg0, %arg0 : i32, i32
}
