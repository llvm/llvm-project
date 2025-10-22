// RUN: mlir-opt -convert-xegpu-to-xevm %s | FileCheck %s

gpu.module @create_nd_tdesc {
  // CHECK-LABEL: gpu.func @create_nd_tdesc
  // CHECK-SAME: %[[ARG0:.*]]: memref<16x32xf32, 1>, %[[ARG1:.*]]: ui64,
  // CHECK-SAME: %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index, %[[ARG5:.*]]: index, %[[ARG6:.*]]: index, %[[ARG7:.*]]: index
  // CHECK-SAME: %[[ARG8:.*]]: memref<?x?xf16>) kernel {
  gpu.func @create_nd_tdesc(%src: memref<16x32xf32, 1>, %ptr: ui64, %shape1: index, %shape2: index,
  %stride1: index, %stride2: index, %offset1: index, %offset2: index, %dyn: memref<?x?xf16>) kernel {
        // Optimized away
        %ptr_tdesc = xegpu.create_nd_tdesc %ptr, shape:[%shape1, %shape2], strides:[%stride1, %stride2]
            : ui64 -> !xegpu.tensor_desc<8x16xf32>
        // CHECK-NEXT: %[[MEMSPACECAST:.*]] = memref.memory_space_cast %[[ARG0]] : memref<16x32xf32, 1> to memref<16x32xf32>
        %srcce = memref.memory_space_cast %src : memref<16x32xf32, 1> to memref<16x32xf32>
        // Optimized away
        %src_tdesc = xegpu.create_nd_tdesc %srcce : memref<16x32xf32> -> !xegpu.tensor_desc<8x16xf32>
        // CHECK-NEXT: %c1 = arith.constant 1 : index
        %c1 = arith.constant 1 : index
        // CHECK-NEXT: %c64 = arith.constant 64 : index
        %size_x = arith.constant 64 : index
        // CHECK-NEXT: %c16 = arith.constant 16 : index
        %BLOCK_DMODEL = arith.constant 16 : index
        // Optimized away
        %dyn_tdesc  = xegpu.create_nd_tdesc %dyn, shape: [%size_x, %BLOCK_DMODEL], strides: [%BLOCK_DMODEL, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<16x16xf16>
        // CHECK-NEXT: gpu.return
        gpu.return
    }
}
