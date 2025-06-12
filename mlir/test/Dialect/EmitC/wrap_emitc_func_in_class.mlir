// RUN: mlir-opt  %s --wrap-emitc-func-in-class

module attributes {tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.schema_version = 3 : i32} {
  emitc.func @main(%arg0: !emitc.array<1xf32> {tf_saved_model.index_path = ["another_feature"]}, %arg1: !emitc.array<1xf32> {tf_saved_model.index_path = ["some_feature"]}, %arg2: !emitc.array<1xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "serving_default_another_feature:0,serving_default_some_feature:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
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
