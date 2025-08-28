// RUN: mlir-opt -convert-xegpu-to-xevm %s | FileCheck %s

gpu.module @create_nd_tdesc {
  // CHECK-LABEL: gpu.func @create_nd_tdesc
  // CHECK-SAME: %[[ARG0:.*]]: memref<16x32xf32, 1>, %[[ARG1:.*]]: ui64,
  // CHECK-SAME: %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index, %[[ARG5:.*]]: index, %[[ARG6:.*]]: index, %[[ARG7:.*]]: index
  gpu.func @create_nd_tdesc(%src: memref<16x32xf32, 1>, %ptr: ui64, %shape1: index, %shape2: index,
  %stride1: index, %stride2: index, %offset1: index, %offset2: index) kernel {
        // CHECK: %[[VAR0:.*]] = index.castu %[[ARG1]] : ui64 to index
        // CHECK: %[[BASE_ADDR:.*]] = arith.index_castui %[[VAR0]] : index to i64
        // CHECK: %[[CST:.*]] = arith.constant dense<0> : vector<8xi32>
        // CHECK: %[[OFFSET_W:.*]] = arith.constant 0 : i32
        // CHECK: %[[OFFSET_H:.*]] = arith.constant 0 : i32
        // CHECK: %[[SHAPE_W:.*]] = arith.index_cast %[[ARG3]] : index to i32
        // CHECK: %[[SHAPE_H:.*]] = arith.index_cast %[[ARG2]] : index to i32
        // CHECK: %[[VAR6:.*]] = vector.bitcast %[[CST]] : vector<8xi32> to vector<4xi64>
        // CHECK: %[[VAR7:.*]] = vector.insert %[[BASE_ADDR]], %[[VAR6]] [0] : i64 into vector<4xi64>
        // CHECK: %[[VAR8:.*]] = vector.bitcast %[[VAR7]] : vector<4xi64> to vector<8xi32>
        // CHECK: %[[VAR9:.*]] = vector.insert %[[SHAPE_W]], %[[VAR8]] [2] : i32 into vector<8xi32>
        // CHECK: %[[VAR10:.*]] = vector.insert %[[SHAPE_H]], %[[VAR9]] [3] : i32 into vector<8xi32>
        // CHECK: %[[VAR11:.*]] = vector.insert %[[OFFSET_W]], %[[VAR10]] [4] : i32 into vector<8xi32>
        // CHECK: %[[VAR12:.*]] = vector.insert %[[OFFSET_H]], %[[VAR11]] [5] : i32 into vector<8xi32>
        %ptr_tdesc = xegpu.create_nd_tdesc %ptr, shape:[%shape1, %shape2], strides:[%stride1, %stride2]
            : ui64 -> !xegpu.tensor_desc<8x16xf32>

        // CHECK: %[[MEMSPACECAST:.*]] = memref.memory_space_cast %[[ARG0]] : memref<16x32xf32, 1> to memref<16x32xf32>
        %srcce = memref.memory_space_cast %src : memref<16x32xf32, 1> to memref<16x32xf32>

        // CHECK: %[[CST_1:.*]] = arith.constant dense<0> : vector<8xi32>
        // CHECK: %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %[[MEMSPACECAST]] : memref<16x32xf32> -> index
        // CHECK: %[[OFFSET_W2:.*]] = arith.constant 0 : i32
        // CHECK: %[[OFFSET_H2:.*]] = arith.constant 0 : i32
        // CHECK: %[[C32_I64:.*]] = arith.constant 32 : i64
        // CHECK: %[[SHAPE_W2:.*]] = arith.trunci %[[C32_I64]] : i64 to i32
        // CHECK: %[[C16_I64:.*]] = arith.constant 16 : i64
        // CHECK: %[[SHAPE_H2:.*]] = arith.trunci %[[C16_I64]] : i64 to i32
        // CHECK: %[[BASE_ADDR2:.*]] = arith.index_castui %[[INTPTR]] : index to i64
        // CHECK: %[[VAR14:.*]] = vector.bitcast %[[CST_1]] : vector<8xi32> to vector<4xi64>
        // CHECK: %[[VAR15:.*]] = vector.insert %[[BASE_ADDR2]], %[[VAR14]] [0] : i64 into vector<4xi64>
        // CHECK: %[[VAR16:.*]] = vector.bitcast %[[VAR15]] : vector<4xi64> to vector<8xi32>
        // CHECK: %[[VAR17:.*]] = vector.insert %[[SHAPE_W2]], %[[VAR16]] [2] : i32 into vector<8xi32>
        // CHECK: %[[VAR18:.*]] = vector.insert %[[SHAPE_H2]], %[[VAR17]] [3] : i32 into vector<8xi32>
        // CHECK: %[[VAR19:.*]] = vector.insert %[[OFFSET_W2]], %[[VAR18]] [4] : i32 into vector<8xi32>
        // CHECK: %[[PAYLOAD:.*]] = vector.insert %[[OFFSET_H2]], %[[VAR19]] [5] : i32 into vector<8xi32>
        %src_tdesc = xegpu.create_nd_tdesc %srcce : memref<16x32xf32> -> !xegpu.tensor_desc<8x16xf32>

        // CHECK: %[[CST_4:.*]] = arith.constant dense<0> : vector<8xi32>
        // CHECK: %[[INTPTR_2:.*]] = memref.extract_aligned_pointer_as_index %[[MEMSPACECAST]] : memref<16x32xf32> -> index
        // CHECK: %[[OFFSET_W3:.*]] = arith.index_cast %[[ARG7]] : index to i32
        // CHECK: %[[OFFSET_H3:.*]] = arith.index_cast %[[ARG6]] : index to i32
        // CHECK: %[[C32_I64_6:.*]] = arith.constant 32 : i64
        // CHECK: %[[SHAPE_W3:.*]] = arith.trunci %[[C32_I64_6]] : i64 to i32
        // CHECK: %[[C16_I64_7:.*]] = arith.constant 16 : i64
        // CHECK: %[[SHAPE_H3:.*]] = arith.trunci %[[C16_I64_7]] : i64 to i32
        // CHECK: %[[BASE_ADDR3:.*]] = arith.index_castui %[[INTPTR_2]] : index to i64
        // CHECK: %[[VAR26:.*]] = vector.bitcast %[[CST_4]] : vector<8xi32> to vector<4xi64>
        // CHECK: %[[VAR27:.*]] = vector.insert %[[BASE_ADDR3]], %[[VAR26]] [0] : i64 into vector<4xi64>
        // CHECK: %[[VAR28:.*]] = vector.bitcast %[[VAR27]] : vector<4xi64> to vector<8xi32>
        // CHECK: %[[VAR29:.*]] = vector.insert %[[SHAPE_W3]], %[[VAR28]] [2] : i32 into vector<8xi32>
        // CHECK: %[[VAR30:.*]] = vector.insert %[[SHAPE_H3]], %[[VAR29]] [3] : i32 into vector<8xi32>
        // CHECK: %[[VAR31:.*]] = vector.insert %[[OFFSET_W3]], %[[VAR30]] [4] : i32 into vector<8xi32>
        // CHECK: %[[VAR32:.*]] = vector.insert %[[OFFSET_H3]], %[[VAR31]] [5] : i32 into vector<8xi32>
        %src_tdesc2 = xegpu.create_nd_tdesc %srcce[%offset1, %offset2] : memref<16x32xf32> -> !xegpu.tensor_desc<8x16xf32>

        // CHECK: %[[C8:.*]] = arith.constant 8 : index
        %c8 = arith.constant 8 : index
        // CHECK: %[[C16:.*]] = arith.constant 16 : index
        %c16 = arith.constant 16 : index
        // CHECK: %[[VAR33:.*]] = arith.index_cast %[[C8]] : index to i32
        // CHECK: %[[OLD_OFFSET_H:.*]] = vector.extract %[[PAYLOAD]][5] : i32 from vector<8xi32>
        // CHECK: %[[NEW_OFFSET_H:.*]] = arith.addi %[[OLD_OFFSET_H]], %[[VAR33]] : i32
        // CHECK: %[[NEW_PAYLOAD:.*]] = vector.insert %[[NEW_OFFSET_H]], %[[PAYLOAD]] [5] : i32 into vector<8xi32>
        // CHECK: %[[VAR37:.*]] = arith.index_cast %[[C16]] : index to i32
        // CHECK: %[[OLD_OFFSET_H:.*]] = vector.extract %[[NEW_PAYLOAD]][4] : i32 from vector<8xi32>
        // CHECK: %[[NEW_OFFSET_H:.*]] = arith.addi %[[OLD_OFFSET_H]], %[[VAR37]] : i32
        // CHECK: %[[FINAL_PAYLOAD:.*]] = vector.insert %[[NEW_OFFSET_H]], %[[NEW_PAYLOAD]] [4] : i32 into vector<8xi32>
        %updated_tdesc = xegpu.update_nd_offset %src_tdesc, [%c8, %c16] : !xegpu.tensor_desc<8x16xf32>
        gpu.return
    }
}
