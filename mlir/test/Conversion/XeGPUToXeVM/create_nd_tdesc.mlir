// RUN: mlir-opt -convert-xegpu-to-xevm %s | FileCheck %s

gpu.module @create_nd_tdesc {
  // CHECK-LABEL: gpu.func @create_nd_tdesc
  // CHECK-SAME: %[[ARG0:.*]]: memref<16x32xf32, 1>, %[[ARG1:.*]]: ui64,
  // CHECK-SAME: %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index, %[[ARG5:.*]]: index, %[[ARG6:.*]]: index, %[[ARG7:.*]]: index
  // CHECK-SAME: %[[DYN:.*]]: memref<?x?xf16>) kernel {
  gpu.func @create_nd_tdesc(%src: memref<16x32xf32, 1>, %ptr: ui64, %shape1: index, %shape2: index,
  %stride1: index, %stride2: index, %offset1: index, %offset2: index, %dyn: memref<?x?xf16>) kernel {
        // CHECK: %[[INTPTR_5:.*]] = memref.extract_aligned_pointer_as_index %[[DYN]] : memref<?x?xf16> -> index
        // CHECK: %[[DYN_ADDR:.*]] = arith.index_castui %[[INTPTR_5]] : index to i64
        // CHECK: %[[VAR0:.*]] = index.castu %[[ARG1]] : ui64 to index
        // CHECK: %[[BASE_ADDR:.*]] = arith.index_castui %[[VAR0]] : index to i64
        // CHECK: %[[CST:.*]] = arith.constant dense<0> : vector<8xi32>
        // CHECK: %[[SHAPE_W:.*]] = arith.index_cast %[[ARG3]] : index to i32
        // CHECK: %[[SHAPE_H:.*]] = arith.index_cast %[[ARG2]] : index to i32
        // CHECK: %[[PITCH:.*]] = arith.index_cast %[[ARG4]] : index to i32
        // CHECK: %[[VAR6:.*]] = vector.bitcast %[[CST]] : vector<8xi32> to vector<4xi64>
        // CHECK: %[[VAR7:.*]] = vector.insert %[[BASE_ADDR]], %[[VAR6]] [0] : i64 into vector<4xi64>
        // CHECK: %[[VAR8:.*]] = vector.bitcast %[[VAR7]] : vector<4xi64> to vector<8xi32>
        // CHECK: %[[VAR9:.*]] = vector.insert %[[SHAPE_W]], %[[VAR8]] [2] : i32 into vector<8xi32>
        // CHECK: %[[VAR10:.*]] = vector.insert %[[SHAPE_H]], %[[VAR9]] [3] : i32 into vector<8xi32>
        // CHECK: %[[VAR11:.*]] = vector.insert %[[PITCH]], %[[VAR10]] [4] : i32 into vector<8xi32>
        %ptr_tdesc = xegpu.create_nd_tdesc %ptr, shape:[%shape1, %shape2], strides:[%stride1, %stride2]
            : ui64 -> !xegpu.tensor_desc<8x16xf32>

        // CHECK: %[[MEMSPACECAST:.*]] = memref.memory_space_cast %[[ARG0]] : memref<16x32xf32, 1> to memref<16x32xf32>
        %srcce = memref.memory_space_cast %src : memref<16x32xf32, 1> to memref<16x32xf32>

        // CHECK: %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %[[MEMSPACECAST]] : memref<16x32xf32> -> index
        // CHECK: %[[BASE_ADDR2:.*]] = arith.index_castui %[[INTPTR]] : index to i64
        // CHECK: %[[CST_1:.*]] = arith.constant dense<0> : vector<8xi32>
        // CHECK: %[[C32_I64:.*]] = arith.constant 32 : i64
        // CHECK: %[[SHAPE_W2:.*]] = arith.trunci %[[C32_I64]] : i64 to i32
        // CHECK: %[[C16_I64:.*]] = arith.constant 16 : i64
        // CHECK: %[[SHAPE_H2:.*]] = arith.trunci %[[C16_I64]] : i64 to i32
        // CHECK: %[[C32_I64_2:.*]] = arith.constant 32 : i64
        // CHECK: %[[PITCH2:.*]] = arith.trunci %[[C32_I64_2]] : i64 to i32
        // CHECK: %[[VAR14:.*]] = vector.bitcast %[[CST_1]] : vector<8xi32> to vector<4xi64>
        // CHECK: %[[VAR15:.*]] = vector.insert %[[BASE_ADDR2_OFFSET:.*]], %[[VAR14]] [0] : i64 into vector<4xi64>
        // CHECK: %[[VAR16:.*]] = vector.bitcast %[[VAR15]] : vector<4xi64> to vector<8xi32>
        // CHECK: %[[VAR17:.*]] = vector.insert %[[SHAPE_W2]], %[[VAR16]] [2] : i32 into vector<8xi32>
        // CHECK: %[[VAR18:.*]] = vector.insert %[[SHAPE_H2]], %[[VAR17]] [3] : i32 into vector<8xi32>
        // CHECK: %[[VAR19:.*]] = vector.insert %[[PITCH2]], %[[VAR18]] [4] : i32 into vector<8xi32>
        %src_tdesc = xegpu.create_nd_tdesc %srcce : memref<16x32xf32> -> !xegpu.tensor_desc<8x16xf32>

        // A dynamic memref uses the bare form; shape/strides come from it.
        // CHECK: %{{.*}}, %{{.*}}, %[[SIZES:.*]]:2, %[[STRIDES:.*]]:2 = memref.extract_strided_metadata %[[DYN]] : memref<?x?xf16>
        // CHECK: %[[CST_3:.*]] = arith.constant dense<0> : vector<8xi32>
        // CHECK: %[[SHAPE_W3:.*]] = arith.index_cast %[[SIZES]]#1 : index to i32
        // CHECK: %[[SHAPE_H3:.*]] = arith.index_cast %[[SIZES]]#0 : index to i32
        // CHECK: %[[PITCH3:.*]] = arith.index_cast %[[STRIDES]]#0 : index to i32
        // CHECK: %[[VAR25:.*]] = vector.bitcast %[[CST_3]] : vector<8xi32> to vector<4xi64>
        // CHECK: %[[VAR26:.*]] = vector.insert %{{.*}}, %[[VAR25]] [0] : i64 into vector<4xi64>
        // CHECK: %[[VAR27:.*]] = vector.bitcast %[[VAR26]] : vector<4xi64> to vector<8xi32>
        // CHECK: %[[VAR28:.*]] = vector.insert %[[SHAPE_W3]], %[[VAR27]] [2] : i32 into vector<8xi32>
        // CHECK: %[[VAR29:.*]] = vector.insert %[[SHAPE_H3]], %[[VAR28]] [3] : i32 into vector<8xi32>
        // CHECK: %[[VAR30:.*]] = vector.insert %[[PITCH3]], %[[VAR29]] [4] : i32 into vector<8xi32>
        %dyn_tdesc  = xegpu.create_nd_tdesc %dyn : memref<?x?xf16> -> !xegpu.tensor_desc<16x16xf16>
        gpu.return
    }

    // Batched (>2D): base_height spans all planes; slot 5 = batch row stride.
    // CHECK-LABEL: gpu.func @create_nd_tdesc_batch_dyn(
    // CHECK-SAME:  %[[SRC:.+]]: memref<?x?x?xf16>
    gpu.func @create_nd_tdesc_batch_dyn(%src: memref<?x?x?xf16>) -> vector<8xi32> {
        // CHECK: %{{.+}}, %{{.+}}, %[[SIZES:.+]]:3, %[[STRIDES:.+]]:3 = memref.extract_strided_metadata %[[SRC]]
        // CHECK: %[[W:.+]] = arith.index_cast %[[SIZES]]#2 : index to i32
        // CHECK: %[[H:.+]] = arith.index_cast %[[SIZES]]#1 : index to i32
        // CHECK: %[[BATCH:.+]] = arith.index_cast %[[SIZES]]#0 : index to i32
        // CHECK: %[[FLAT_H:.+]] = arith.muli %[[H]], %[[BATCH]] : i32
        // CHECK: %[[PITCH:.+]] = arith.index_cast %[[STRIDES]]#1 : index to i32
        // CHECK: %[[P2:.+]] = vector.insert %[[W]], %{{.+}} [2] : i32 into vector<8xi32>
        // CHECK: %[[P3:.+]] = vector.insert %[[FLAT_H]], %[[P2]] [3] : i32 into vector<8xi32>
        // CHECK: %[[P4:.+]] = vector.insert %[[PITCH]], %[[P3]] [4] : i32 into vector<8xi32>
        // CHECK: %[[LS0:.+]] = arith.index_cast %[[STRIDES]]#0 : index to i32
        // CHECK: %[[ROWS0:.+]] = arith.divui %[[LS0]], %[[PITCH]] : i32
        // CHECK: vector.insert %[[ROWS0]], %[[P4]] [5] : i32 into vector<8xi32>
        %t = xegpu.create_nd_tdesc %src : memref<?x?x?xf16> -> !xegpu.tensor_desc<1x8x16xf16>
        %c = builtin.unrealized_conversion_cast %t : !xegpu.tensor_desc<1x8x16xf16> to vector<8xi32>
        gpu.return %c : vector<8xi32>
    }
}
