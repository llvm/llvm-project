// RUN: mlir-opt --wrap-emitc-func-in-class='named-attribute=tf_saved_model.index_path' %s | FileCheck %s

module attributes {tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.schema_version = 3 : i32} {
  emitc.func @Model(%arg0: !emitc.array<1xf32> {tf_saved_model.index_path = ["another_feature"]}, %arg1: !emitc.array<1xf32> {tf_saved_model.index_path = ["some_feature"]}, %arg2: !emitc.array<1xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "serving_default_another_feature:0,serving_default_some_feature:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
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

// CHECK: module attributes {tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.schema_version = 3 : i32} {
// CHECK:   emitc.class @MyModelClass {
// CHECK:     emitc.field @another_feature : !emitc.array<1xf32> = {tf_saved_model.index_path = ["another_feature"]}
// CHECK:     emitc.field @some_feature : !emitc.array<1xf32> = {tf_saved_model.index_path = ["some_feature"]}
// CHECK:     emitc.field @output_0 : !emitc.array<1xf32> = {tf_saved_model.index_path = ["output_0"]}
// CHECK:     emitc.func @execute() {
// CHECK:       %{{[0-9]+}} = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
// CHECK:       %{{[0-9]+}} = get_field @another_feature : !emitc.array<1xf32>
// CHECK:       %{{[0-9]+}} = get_field @some_feature : !emitc.array<1xf32>
// CHECK:       %{{[0-9]+}} = get_field @output_0 : !emitc.array<1xf32>
// CHECK:       %{{[0-9]+}} = subscript %{{[0-9]+}}[%{{[0-9]+}}] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
// CHECK:       %{{[0-9]+}} = load %{{[0-9]+}} : <f32>
// CHECK:       %{{[0-9]+}} = subscript %{{[0-9]+}}[%{{[0-9]+}}] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
// CHECK:       %{{[0-9]+}} = load %{{[0-9]+}} : <f32>
// CHECK:       %{{[0-9]+}} = add %{{[0-9]+}}, %{{[0-9]+}} : (f32, f32) -> f32
// CHECK:       %{{[0-9]+}} = subscript %{{[0-9]+}}[%{{[0-9]+}}] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
// CHECK:       assign %{{[0-9]+}} : f32 to %{{[0-9]+}} : <f32>
// CHECK:       return
// CHECK:     }
// CHECK:   }
// CHECK: }
