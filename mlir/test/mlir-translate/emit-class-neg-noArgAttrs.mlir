/// The function has no argument attributes
// RUN: not mlir-translate --mlir-to-cpp --emit-class=true --class-name=ArgAttrs --field-name-attribute=tf_saved_model.index_path %s | FileCheck %s

module attributes {tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.schema_version = 3 : i32} {
  emitc.func @foo(%arg0 : i32) {
    emitc.call_opaque "bar" (%arg0) : (i32) -> ()
    emitc.return
  }
}

// CHECK: class ArgAttrs final {
// CHECK-NEXT: public: 
// CHECK-NEXT:   int32_t v1;
// CHECK-EMPTY: 
// CHECK-NEXT:   std::map<std::string, char*> _buffer_map {
