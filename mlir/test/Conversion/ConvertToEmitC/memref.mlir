// RUN: mlir-opt -convert-to-emitc -split-input-file %s | FileCheck %s

// CHECK-LABEL: emitc.func @test_memref_alloc()
func.func @test_memref_alloc() {
  // CHECK: %[[SIZEOF:.*]] = call_opaque "sizeof"() <{args = [!emitc.opaque<"TestElementT">]}> : () -> !emitc.size_t
  // CHECK: %[[C10:.*]] = "emitc.constant"() <{value = 10 : index}> : () -> index
  // CHECK: %[[BYTES:.*]] = mul %[[SIZEOF]], %[[C10]] : (!emitc.size_t, index) -> !emitc.size_t
  // CHECK: %[[MALLOC:.*]] = call_opaque "malloc"(%[[BYTES]]) : (!emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">>
  // CHECK: %[[CAST:.*]] = cast %[[MALLOC]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<!emitc.opaque<"TestElementT">>
  %0 = memref.alloc() : memref<10x!test.memref_element>
  return
}

// -----

// CHECK-LABEL:   emitc.func @test_memref_copy(
// CHECK-SAME:      %[[ARG0:.*]]: !emitc.array<10x!emitc.opaque<"TestElementT">>,
// CHECK-SAME:      %[[ARG1:.*]]: !emitc.array<10x!emitc.opaque<"TestElementT">>) {
func.func @test_memref_copy(%arg0: memref<10x!test.memref_element>, %arg1: memref<10x!test.memref_element>) {
  // CHECK:           %[[C0_0:.*]] = "emitc.constant"() <{value = 0 : index}> : () -> index
  // CHECK:           %[[SUB_0:.*]] = subscript %[[ARG0]][%[[C0_0]]] : (!emitc.array<10x!emitc.opaque<"TestElementT">>, index) -> !emitc.lvalue<!emitc.opaque<"TestElementT">>
  // CHECK:           %[[ADDR_0:.*]] = address_of %[[SUB_0]] : !emitc.lvalue<!emitc.opaque<"TestElementT">>
  // CHECK:           %[[C0_1:.*]] = "emitc.constant"() <{value = 0 : index}> : () -> index
  // CHECK:           %[[SUB_1:.*]] = subscript %[[ARG1]][%[[C0_1]]] : (!emitc.array<10x!emitc.opaque<"TestElementT">>, index) -> !emitc.lvalue<!emitc.opaque<"TestElementT">>
  // CHECK:           %[[ADDR_1:.*]] = address_of %[[SUB_1]] : !emitc.lvalue<!emitc.opaque<"TestElementT">>
  // CHECK:           %[[SIZEOF:.*]] = call_opaque "sizeof"() <{args = [!emitc.opaque<"TestElementT">]}> : () -> !emitc.size_t
  // CHECK:           %[[C10:.*]] = "emitc.constant"() <{value = 10 : index}> : () -> index
  // CHECK:           %[[BYTES:.*]] = mul %[[SIZEOF]], %[[C10]] : (!emitc.size_t, index) -> !emitc.size_t
  // CHECK:           call_opaque "memcpy"(%[[ADDR_1]], %[[ADDR_0]], %[[BYTES]]) : (!emitc.ptr<!emitc.opaque<"TestElementT">>, !emitc.ptr<!emitc.opaque<"TestElementT">>, !emitc.size_t) -> ()
  memref.copy %arg0, %arg1 : memref<10x!test.memref_element> to memref<10x!test.memref_element>
  return
}
