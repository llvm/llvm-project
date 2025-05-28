/// An external function - has no body
// RUN: not mlir-translate --mlir-to-cpp --emit-class=true --class-name=MyAdder --field-name-attribute=tf_saved_model.index_path %s 2>&1 | FileCheck %s

module attributes {tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.metadata = {CONVERSION_METADATA = "\10\00\00\00\00\00\00\00\08\00\0E\00\08\00\04\00\08\00\00\00\10\00\00\00$\00\00\00\00\00\06\00\08\00\04\00\06\00\00\00\04\00\00\00\00\00\00\00\0C\00\18\00\14\00\10\00\0C\00\04\00\0C\00\00\00\A6\03|\7Frm\F2\17\01\00\00\00\02\00\00\00\04\00\00\00\06\00\00\002.19.0\00\00", min_runtime_version = "1.5.0\00\00\00\00\00\00\00\00\00\00\00"}, tfl.schema_version = 3 : i32} {
  emitc.func private @extern_func(i32) attributes {specifiers = ["extern"]}
}

// CHECK: Warning: Cannot process external function 'extern_func'. It lacks a body, and attempting to continue would lead to errors due to missing argument details.