// RUN: mlir-translate --mlir-to-cpp --emit-class=true --class-name=MyAdder --field-name-attribute=tf_saved_model.index_path %s | FileCheck %s  

module attributes {tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.metadata = {CONVERSION_METADATA = "\10\00\00\00\00\00\00\00\08\00\0E\00\08\00\04\00\08\00\00\00\10\00\00\00$\00\00\00\00\00\06\00\08\00\04\00\06\00\00\00\04\00\00\00\00\00\00\00\0C\00\18\00\14\00\10\00\0C\00\04\00\0C\00\00\00\A6\03|\7Frm\F2\17\01\00\00\00\02\00\00\00\04\00\00\00\06\00\00\002.19.0\00\00", min_runtime_version = "1.5.0\00\00\00\00\00\00\00\00\00\00\00"}, tfl.schema_version = 3 : i32} {
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

// CHECK: class MyAdder final {
// CHECK-NEXT: public:
// CHECK-NEXT:   float v1[1];
// CHECK-NEXT:   float v2[1];
// CHECK-NEXT:   float v3[1];
// CHECK-EMPTY:
// CHECK-NEXT:   std::map<std::string, char*> _buffer_map { { "another_feature", reinterpret_cast<char*>(v1) }, { "some_feature", reinterpret_cast<char*>(v2) }, { "output_0", reinterpret_cast<char*>(v3) }, };
// CHECK-NEXT:   char* getBufferForName(const std::string& name) const {
// CHECK-NEXT:      auto it = _buffer_map.find(name);
// CHECK-NEXT:      return (it == _buffer_map.end()) ? nullptr : it->second;
// CHECK-NEXT:   }
// CHECK-EMPTY:
// CHECK-NEXT: void main() {
// CHECK-NEXT:     size_t v4 = 0;
// CHECK-NEXT:     float v5 = v2[v4];
// CHECK-NEXT:     float v6 = v1[v4];
// CHECK-NEXT:     float v7 = v5 + v6;
// CHECK-NEXT:     v3[v4] = v7;
// CHECK-NEXT:     return;
// CHECK-NEXT:  }
// CHECK-NEXT: };

