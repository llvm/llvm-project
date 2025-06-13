// RUN: mlir-opt --wrap-emitc-func-in-class='named-attribute=tf_saved_model.index_path' %s 2>&1 | FileCheck %s

emitc.func @foo(%arg0 : i32) {
  emitc.call_opaque "bar" (%arg0) : (i32) -> ()
  emitc.return
}

// CHECK: error: 'emitc.func' op arguments should have attributes so we can initialize class fields.
