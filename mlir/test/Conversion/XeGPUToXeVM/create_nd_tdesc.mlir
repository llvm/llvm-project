// RUN: mlir-opt -convert-xegpu-to-xevm %s | FileCheck %s

gpu.module @create_nd_tdesc {
  // CHECK-LABEL: gpu.func @create_nd_tdesc
  // CHECK-SAME: %[[ARG0:.*]]: memref<8x16xf32, 1>, %[[ARG1:.*]]: ui64
  // CHECK-SAME: %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index, %[[ARG5:.*]]: index
  gpu.func @create_nd_tdesc(%src: memref<8x16xf32, 1>, %ptr: ui64, %shape1: index, %shape2: index,
       %stride1: index, %stride2: index) kernel {
        // CHECK: %[[VAR0:.*]] = index.castu %[[ARG1]] : ui64 to index
        // CHECK: %[[VAR1:.*]] = arith.index_castui %[[VAR0]] : index to i64
        // CHECK: %[[CST:.*]] = arith.constant dense<0> : vector<8xi32>
        // CHECK: %[[C0_I32:.*]] = arith.constant 0 : i32
        // CHECK: %[[C0_I32_0:.*]] = arith.constant 0 : i32
        // CHECK: %[[VAR3:.*]] = arith.index_cast %[[ARG3]] : index to i32
        // CHECK: %[[VAR5:.*]] = arith.index_cast %[[ARG2]] : index to i32
        // CHECK: %[[VAR6:.*]] = vector.bitcast %[[CST]] : vector<8xi32> to vector<4xi64>
        // CHECK: %[[VAR7:.*]] = vector.insert %[[VAR1]], %[[VAR6]] [0] : i64 into vector<4xi64>
        // CHECK: %[[VAR8:.*]] = vector.bitcast %[[VAR7]] : vector<4xi64> to vector<8xi32>
        // CHECK: %[[VAR9:.*]] = vector.insert %[[VAR3]], %[[VAR8]] [2] : i32 into vector<8xi32>
        // CHECK: %[[VAR10:.*]] = vector.insert %[[VAR5]], %[[VAR9]] [3] : i32 into vector<8xi32>
        // CHECK: %[[VAR11:.*]] = vector.insert %[[C0_I32]], %[[VAR10]] [4] : i32 into vector<8xi32>
        // CHECK: %[[VAR12:.*]] = vector.insert %[[C0_I32_0]], %[[VAR11]] [5] : i32 into vector<8xi32>
        %ptr_tdesc = xegpu.create_nd_tdesc %ptr, shape:[%shape1, %shape2], strides:[%stride1, %stride2]
            : ui64 -> !xegpu.tensor_desc<8x16xf32>

        // CHECK: %[[MEMSPACECAST:.*]] = memref.memory_space_cast %[[ARG0]] : memref<8x16xf32, 1> to memref<8x16xf32>
        %srcce = memref.memory_space_cast %src : memref<8x16xf32, 1> to memref<8x16xf32>

        // CHECK: %[[CST_1:.*]] = arith.constant dense<0> : vector<8xi32>
        // CHECK: %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %[[MEMSPACECAST]] : memref<8x16xf32> -> index
        // CHECK: %[[C0_I32_2:.*]] = arith.constant 0 : i32
        // CHECK: %[[C0_I32_3:.*]] = arith.constant 0 : i32
        // CHECK: %[[C16_I64:.*]] = arith.constant 16 : i64
        // CHECK: %[[C16_I32:.*]] = arith.trunci %c16_i64 : i64 to i32
        // CHECK: %[[C8_I64:.*]] = arith.constant 8 : i64
        // CHECK: %[[C8_I32:.*]] = arith.trunci %c8_i64 : i64 to i32
        // CHECK: %[[VAR13:.*]] = arith.index_castui %[[INTPTR]] : index to i64
        // CHECK: %[[VAR14:.*]] = vector.bitcast %[[CST_1]] : vector<8xi32> to vector<4xi64>
        // CHECK: %[[VAR15:.*]] = vector.insert %[[VAR13]], %[[VAR14]] [0] : i64 into vector<4xi64>
        // CHECK: %[[VAR16:.*]] = vector.bitcast %[[VAR15]] : vector<4xi64> to vector<8xi32>
        // CHECK: %[[VAR17:.*]] = vector.insert %[[C16_I32]], %[[VAR16]] [2] : i32 into vector<8xi32>
        // CHECK: %[[VAR18:.*]] = vector.insert %[[C8_I32]], %[[VAR17]] [3] : i32 into vector<8xi32>
        // CHECK: %[[VAR19:.*]] = vector.insert %[[C0_I32_2]], %[[VAR18]] [4] : i32 into vector<8xi32>
        // CHECK: %[[VAR20:.*]] = vector.insert %[[C0_I32_3]], %[[VAR19]] [5] : i32 into vector<8xi32>
        %src_tdesc = xegpu.create_nd_tdesc %srcce : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
        gpu.return
    }
}
