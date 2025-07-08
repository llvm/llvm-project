// RUN: mlir-opt --wrap-emitc-func-in-class='named-attribute=emitc.name_hint' %s | FileCheck %s

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
// CHECK-NEXT:   emitc.class @modelClass {
// CHECK-NEXT:     emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.name_hint = "another_feature"}
// CHECK-NEXT:     emitc.field @fieldName1 : !emitc.array<1xf32>  {emitc.name_hint = "some_feature"}
// CHECK-NEXT:     emitc.field @fieldName2 : !emitc.array<1xf32>  {emitc.name_hint = "output_0"}
// CHECK-NEXT:     emitc.func @execute() {
// CHECK-NEXT:       get_field @fieldName0 : !emitc.array<1xf32>
// CHECK-NEXT:       get_field @fieldName1 : !emitc.array<1xf32>
// CHECK-NEXT:       get_field @fieldName2 : !emitc.array<1xf32>
// CHECK-NEXT:       "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
// CHECK-NEXT:       subscript {{.*}}[{{.*}}] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
// CHECK-NEXT:       load {{.*}} : <f32>
// CHECK-NEXT:       subscript {{.*}}[{{.*}}] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
// CHECK-NEXT:       load {{.*}} : <f32>
// CHECK-NEXT:       add {{.*}}, {{.*}} : (f32, f32) -> f32
// CHECK-NEXT:       subscript {{.*}}[{{.*}}] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
// CHECK-NEXT:       assign {{.*}} : f32 to {{.*}} : <f32>
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
