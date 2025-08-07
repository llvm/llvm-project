// RUN: mlir-opt %s -wrap-emitc-func-in-class -split-input-file | FileCheck %s

emitc.func @foo(%arg0 : !emitc.array<1xf32>) {
  emitc.call_opaque "bar" (%arg0) : (!emitc.array<1xf32>) -> ()
  emitc.return
}

// CHECK: module {
// CHECK:   emitc.class @fooClass {
// CHECK:     emitc.field @fieldName0 : !emitc.array<1xf32>
// CHECK:     emitc.func @execute() {
// CHECK:       %0 = get_field @fieldName0 : !emitc.array<1xf32>
// CHECK:       call_opaque "bar"(%0) : (!emitc.array<1xf32>) -> ()
// CHECK:       return
// CHECK:     }
// CHECK:   }
// CHECK: }

// -----

module attributes { } {
  emitc.func @model(%arg0: !emitc.array<1xf32> {emitc.name_hint = "another_feature"},
   %arg1: !emitc.array<1xf32> {emitc.name_hint = "some_feature"},
   %arg2: !emitc.array<1xf32> {emitc.name_hint = "output_0"}) attributes { } {
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
// CHECK:   emitc.class @modelClass {
// CHECK:     emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.name_hint = "another_feature"}
// CHECK:     emitc.field @fieldName1 : !emitc.array<1xf32>  {emitc.name_hint = "some_feature"}
// CHECK:     emitc.field @fieldName2 : !emitc.array<1xf32>  {emitc.name_hint = "output_0"}
// CHECK:     emitc.func @execute() {
// CHECK:       get_field @fieldName0 : !emitc.array<1xf32>
// CHECK:       get_field @fieldName1 : !emitc.array<1xf32>
// CHECK:       get_field @fieldName2 : !emitc.array<1xf32>
// CHECK:       "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
// CHECK:       subscript {{.*}}[{{.*}}] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
// CHECK:       load {{.*}} : <f32>
// CHECK:       subscript {{.*}}[{{.*}}] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
// CHECK:       load {{.*}} : <f32>
// CHECK:       add {{.*}}, {{.*}} : (f32, f32) -> f32
// CHECK:       subscript {{.*}}[{{.*}}] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
// CHECK:       assign {{.*}} : f32 to {{.*}} : <f32>
// CHECK:       return
// CHECK:     }
// CHECK:   }
// CHECK: }
