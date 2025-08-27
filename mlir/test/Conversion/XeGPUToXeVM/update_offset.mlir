// RUN: mlir-opt -convert-xegpu-to-xevm %s | FileCheck %s

gpu.module @update_offset {
  // CHECK-LABEL: gpu.func @update_offset
  // CHECK-SAME: %[[ARG0:.*]]: memref<128xf32>
  gpu.func @update_offset(%src: memref<128xf32>) kernel {
    // CHECK: %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %[[ARG0]] : memref<128xf32> -> index
    // CHECK: %[[VAR2:.*]] = arith.index_castui %[[INTPTR]] : index to i64
    // CHECK: %[[CST:.*]] = arith.constant dense<0> : vector<1xindex>
    %offset = arith.constant dense<0> : vector<1xindex>
    // CHECK: %[[VAR0:.*]] = vector.extract %[[CST]][0] : index from vector<1xindex>
    // CHECK: %[[VAR1:.*]] = arith.index_castui %[[VAR0]] : index to i64
    // CHECK: %[[C4_I64:.*]] = arith.constant 4 : i64
    // CHECK: %[[VAR3:.*]] = arith.muli %[[VAR1]], %[[C4_I64]] : i64
    // CHECK: %[[VAR4:.*]] = arith.addi %[[VAR2]], %[[VAR3]] : i64
    %src_tdesc = xegpu.create_tdesc %src, %offset : memref<128xf32>, vector<1xindex>
        -> !xegpu.tensor_desc<1xf32, #xegpu.scatter_tdesc_attr<>>
    // CHECK: %[[C4_I64_0:.*]] = arith.constant 4 : i64
    // CHECK: %[[VAR5:.*]] = arith.muli %[[VAR1]], %[[C4_I64_0]] : i64
    // CHECK: %[[VAR6:.*]] = arith.addi %[[VAR4]], %[[VAR5]] : i64
    %new_tdesc = xegpu.update_offset %src_tdesc, %offset : !xegpu.tensor_desc<1xf32, #xegpu.scatter_tdesc_attr<>>
        , vector<1xindex>
    gpu.return
  }
}
