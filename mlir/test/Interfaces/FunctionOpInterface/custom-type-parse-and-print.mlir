// RUN: mlir-opt %s | FileCheck %s

//      CHECK: test.custom_type_format_func @single_arg_no_return(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: type:f32) {
// CHECK-NEXT: }
test.custom_type_format_func @single_arg_no_return(%arg0 : type:f32) {}

//      CHECK: test.custom_type_format_func @multiple_arg_multiple_return(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: type:f32
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]: type:f32)
// CHECK-SAME: -> (type:f32, type:f32) {
// CHECK-NEXT: }
test.custom_type_format_func @multiple_arg_multiple_return(
    %arg0 : type:f32, %arg1 : type:f32)
    -> (type:f32, type:f32) {}

//      CHECK: test.custom_type_format_func @no_block
// CHECK-SAME: (type:f32, type:f32)
// CHECK-SAME: -> (type:f32, type:f32)
test.custom_type_format_func @no_block(%arg0 : type:f32, %arg1 : type:f32)
    -> (type:f32, type:f32)

//      CHECK: test.custom_type_format_func @one_return
// CHECK-SAME: -> type:f32
test.custom_type_format_func @one_return()
    -> type:f32
