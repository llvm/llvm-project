// RUN: mlir-opt -convert-memref-to-emitc %s -split-input-file | FileCheck %s

func.func @copying(%arg0 : memref<2x4xf32>) {
  memref.copy %arg0, %arg0 : memref<2x4xf32> to memref<2x4xf32>
  return
}

// func.func @copying_memcpy(%arg_0: !emitc.ptr<f32>) {
//   %size = "emitc.constant"() <{value = 8 : index}> :() -> index
//   %element_size = "emitc.constant"() <{value = 4 : index}> :() -> index
//   %total_bytes = emitc.mul %size, %element_size : (index, index) -> index
  
//   emitc.call_opaque "memcpy"(%arg_0, %arg_0, %total_bytes) : (!emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()
//   return
// }

// CHECK-LABEL: copying_memcpy
// CHECK-SAME: %arg_0: !emitc.ptr<f32>
// CHECK-NEXT: %size = "emitc.constant"() <{value = 8 : index}> :() -> index
// CHECK-NEXT: %element_size = "emitc.constant"() <{value = 4 : index}> :() -> index
// CHECK-NEXT: %total_bytes = emitc.mul %size, %element_size : (index, index) -> index
// CHECK-NEXT: emitc.call_opaque "memcpy"
// CHECK-SAME: (%arg_0, %arg_0, %total_bytes)
// CHECK-NEXT: : (!emitc.ptr<f32>, !emitc.ptr<f32>, index) -> ()
// CHECK-NEXT: return