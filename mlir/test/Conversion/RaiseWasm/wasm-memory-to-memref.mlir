// RUN: mlir-opt %s --raise-wasm-mlir | FileCheck %s

"wasmssa.memory"() <{limits = !wasmssa<limit[17:]>, sym_name = "mem_0", sym_visibility = "nested"}> : () -> ()

// CHECK-LABEL:   memref.global "nested" @mem_0 : memref<1xmemref<?xi8>>

// CHECK-LABEL:   func.func @"mem_0::initializer"() attributes {initializer} {
// CHECK:           %[[VAL_0:.*]] = memref.get_global @mem_0 : memref<1xmemref<?xi8>>
// CHECK:           %[[VAL_1:.*]] = memref.alloc() : memref<17xi8>
// CHECK:           %[[VAL_2:.*]] = memref.cast %[[VAL_1]] : memref<17xi8> to memref<?xi8>
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           memref.store %[[VAL_2]], %[[VAL_0]]{{\[}}%[[VAL_3]]] : memref<1xmemref<?xi8>>
// CHECK:           return
// CHECK:         func.call @"mem_0::initializer"() : () -> ()
