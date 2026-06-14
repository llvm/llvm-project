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
