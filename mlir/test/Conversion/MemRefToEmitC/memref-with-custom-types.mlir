// RUN: mlir-opt -memref-to-emitc -split-input-file %s | FileCheck %s

// CHECK-LABEL: emitc.func @alloc_with_custom_element_type()
func.func @alloc_with_custom_element_type() {
  // CHECK:       call_opaque "sizeof"() <{args = [!emitc.opaque<"TestElementT">]}> : () -> !emitc.size_t
  // CHECK:       cast
  // CHECK-SAME:  !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<!emitc.opaque<"TestElementT">>
  %0 = memref.alloc() : memref<10x!test.memref_element>
  return
}

// -----

// CHECK-LABEL:   emitc.func @copy_with_custom_element_type(
func.func @copy_with_custom_element_type(%arg0: memref<10x!test.memref_element>, %arg1: memref<10x!test.memref_element>) {
  // CHECK:           call_opaque "memcpy"
  // CHECK-SAME:      (!emitc.ptr<!emitc.opaque<"TestElementT">>, !emitc.ptr<!emitc.opaque<"TestElementT">>, !emitc.size_t) -> ()
  memref.copy %arg0, %arg1 : memref<10x!test.memref_element> to memref<10x!test.memref_element>
  return
}
