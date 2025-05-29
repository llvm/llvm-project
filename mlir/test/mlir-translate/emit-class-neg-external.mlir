/// An external function - has no body
// RUN: mlir-translate --mlir-to-cpp --emit-class=true --class-name=MyAdder --field-name-attribute=tf_saved_model.index_path %s | FileCheck %s

module attributes {tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.schema_version = 3 : i32} {
  emitc.func private @extern_func(i32) attributes {specifiers = ["extern"]}
}

// CHECK: Warning: Cannot process external function 'extern_func'. This functionOp lacks a body so we will skip over it.
