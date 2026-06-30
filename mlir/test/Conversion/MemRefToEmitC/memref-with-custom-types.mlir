// Note that we use `-convert-to-emitc` instead of `-convert-memref-to-emitc`
// to include the custom type converter registration for the test dialect.

// RUN: mlir-opt -convert-to-emitc -split-input-file %s | FileCheck %s

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

// -----

// CHECK-LABEL:   emitc.func @store_with_custom_element_type(
func.func @store_with_custom_element_type(%v : !test.memref_element, %i: index) {
  %alloc = memref.alloc() : memref<4x!test.memref_element>
  // CHECK:           assign
  // CHECK-SAME:      <!emitc.opaque<"TestElementT">>
  memref.store %v, %alloc[%i] : memref<4x!test.memref_element>
  return
}

// -----

// CHECK-LABEL:   emitc.func @load_with_custom_element_type(
func.func @load_with_custom_element_type(%i: index) -> !test.memref_element {
  %alloc = memref.alloc() : memref<4x!test.memref_element>
  // CHECK:           load
  // CHECK-SAME:      <!emitc.opaque<"TestElementT">>
  %v = memref.load %alloc[%i] : memref<4x!test.memref_element>
  return %v : !test.memref_element
}

