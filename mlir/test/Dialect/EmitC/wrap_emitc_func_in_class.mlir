// RUN: mlir-opt --wrap-emitc-func-in-class='named-attribute=emitc.opaque' %s | FileCheck %s

module attributes { } {
  emitc.func @model(%arg0: !emitc.array<1xf32> {emitc.opaque = ["another_feature"]}, %arg1: !emitc.array<1xf32> {emitc.opaque = ["some_feature"]}, %arg2: !emitc.array<1xf32> {emitc.opaque = ["output_0"]}) attributes { } {
    %0 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
    %1 = subscript %arg1[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
    %2 = load %1 : <f32>
    %3 = subscript %arg0[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
    %4 = load %3 : <f32>
    %5 = add %2, %4 : (f32, f32) -> f32
    %6 = subscript %arg2[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
    assign %5 : f32 to %6 : <f32>
    return
  }
}


// CHECK: module {
// CHECK-NEXT:   emitc.class @modelClass {
// CHECK-NEXT:     emitc.field @another_feature : !emitc.array<1xf32> = {emitc.opaque = ["another_feature"]}
// CHECK-NEXT:     emitc.field @some_feature : !emitc.array<1xf32> = {emitc.opaque = ["some_feature"]}
// CHECK-NEXT:     emitc.field @output_0 : !emitc.array<1xf32> = {emitc.opaque = ["output_0"]}
// CHECK-NEXT:     emitc.func @execute() {
// CHECK-NEXT:       %{{[0-9]+}} = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
// CHECK-NEXT:       %{{[0-9]+}} = get_field @another_feature : !emitc.array<1xf32>
// CHECK-NEXT:       %{{[0-9]+}} = get_field @some_feature : !emitc.array<1xf32>
// CHECK-NEXT:       %{{[0-9]+}} = get_field @output_0 : !emitc.array<1xf32>
// CHECK-NEXT:       %{{[0-9]+}} = subscript %{{[0-9]+}}[%{{[0-9]+}}] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
// CHECK-NEXT:       %{{[0-9]+}} = load %{{[0-9]+}} : <f32>
// CHECK-NEXT:       %{{[0-9]+}} = subscript %{{[0-9]+}}[%{{[0-9]+}}] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
// CHECK-NEXT:       %{{[0-9]+}} = load %{{[0-9]+}} : <f32>
// CHECK-NEXT:       %{{[0-9]+}} = add %{{[0-9]+}}, %{{[0-9]+}} : (f32, f32) -> f32
// CHECK-NEXT:       %{{[0-9]+}} = subscript %{{[0-9]+}}[%{{[0-9]+}}] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
// CHECK-NEXT:       assign %{{[0-9]+}} : f32 to %{{[0-9]+}} : <f32>
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

