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
        // CHECK: %[[PITCH:.*]] = arith.index_cast %[[ARG4]] : index to i32
        // CHECK: %[[SHAPE_H:.*]] = arith.index_cast %[[ARG2]] : index to i32
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
        // CHECK: %[[C32_I64_2:.*]] = arith.constant 32 : i64
        // CHECK: %[[PITCH2:.*]] = arith.trunci %[[C32_I64_2]] : i64 to i32
        // CHECK: %[[C16_I64:.*]] = arith.constant 16 : i64
        // CHECK: %[[SHAPE_H2:.*]] = arith.trunci %[[C16_I64]] : i64 to i32
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
        // CHECK: %[[PITCH3:.*]] = arith.index_cast %[[STRIDES]]#0 : index to i32
        // CHECK: %[[SHAPE_H3:.*]] = arith.index_cast %[[SIZES]]#0 : index to i32
        // CHECK: %[[VAR25:.*]] = vector.bitcast %[[CST_3]] : vector<8xi32> to vector<4xi64>
        // CHECK: %[[VAR26:.*]] = vector.insert %{{.*}}, %[[VAR25]] [0] : i64 into vector<4xi64>
        // CHECK: %[[VAR27:.*]] = vector.bitcast %[[VAR26]] : vector<4xi64> to vector<8xi32>
        // CHECK: %[[VAR28:.*]] = vector.insert %[[SHAPE_W3]], %[[VAR27]] [2] : i32 into vector<8xi32>
        // CHECK: %[[VAR29:.*]] = vector.insert %[[SHAPE_H3]], %[[VAR28]] [3] : i32 into vector<8xi32>
        // CHECK: %[[VAR30:.*]] = vector.insert %[[PITCH3]], %[[VAR29]] [4] : i32 into vector<8xi32>
        %dyn_tdesc  = xegpu.create_nd_tdesc %dyn : memref<?x?xf16> -> !xegpu.tensor_desc<16x16xf16>
        gpu.return
    }

    // Batched (>2D) with two leading dims, fully dynamic sizes and strides:
    // base_height is the row extent the batch dims reach,
    //   size[2] + (size[0] - 1) * rows0 + (size[1] - 1) * rows1
    // with rows_d = stride[d] / pitch; slots 5 and 6 are the batch row strides.
    // CHECK-LABEL: gpu.func @create_nd_tdesc_batch_dyn(
    // CHECK-SAME:  %[[SRC:.+]]: memref<?x?x?x?xf16>
    gpu.func @create_nd_tdesc_batch_dyn(%src: memref<?x?x?x?xf16>) -> vector<8xi32> {
        // CHECK: %{{.+}}, %{{.+}}, %[[SIZES:.+]]:4, %[[STRIDES:.+]]:4 = memref.extract_strided_metadata %[[SRC]]
        // CHECK: %[[W:.+]] = arith.index_cast %[[SIZES]]#3 : index to i32
        // CHECK: %[[PITCH:.+]] = arith.index_cast %[[STRIDES]]#2 : index to i32
        // CHECK: %[[LS0:.+]] = arith.index_cast %[[STRIDES]]#0 : index to i32
        // CHECK: %[[ROWS0:.+]] = arith.divui %[[LS0]], %[[PITCH]] : i32
        // CHECK: %[[LS1:.+]] = arith.index_cast %[[STRIDES]]#1 : index to i32
        // CHECK: %[[ROWS1:.+]] = arith.divui %[[LS1]], %[[PITCH]] : i32
        // CHECK: %[[H:.+]] = arith.index_cast %[[SIZES]]#2 : index to i32
        // CHECK: %[[C1I32:.+]] = arith.constant 1 : i32
        // CHECK: %[[BATCH0:.+]] = arith.index_cast %[[SIZES]]#0 : index to i32
        // CHECK: %[[BATCH0M1:.+]] = arith.subi %[[BATCH0]], %[[C1I32]] : i32
        // CHECK: %[[BATCH0_ROWS:.+]] = arith.muli %[[BATCH0M1]], %[[ROWS0]] : i32
        // CHECK: %[[FLAT_H0:.+]] = arith.addi %[[H]], %[[BATCH0_ROWS]] : i32
        // CHECK: %[[BATCH1:.+]] = arith.index_cast %[[SIZES]]#1 : index to i32
        // CHECK: %[[BATCH1M1:.+]] = arith.subi %[[BATCH1]], %[[C1I32]] : i32
        // CHECK: %[[BATCH1_ROWS:.+]] = arith.muli %[[BATCH1M1]], %[[ROWS1]] : i32
        // CHECK: %[[FLAT_H:.+]] = arith.addi %[[FLAT_H0]], %[[BATCH1_ROWS]] : i32
        // CHECK: %[[P2:.+]] = vector.insert %[[W]], %{{.+}} [2] : i32 into vector<8xi32>
        // CHECK: %[[P3:.+]] = vector.insert %[[FLAT_H]], %[[P2]] [3] : i32 into vector<8xi32>
        // CHECK: %[[P4:.+]] = vector.insert %[[PITCH]], %[[P3]] [4] : i32 into vector<8xi32>
        // CHECK: %[[P5:.+]] = vector.insert %[[ROWS0]], %[[P4]] [5] : i32 into vector<8xi32>
        // CHECK: vector.insert %[[ROWS1]], %[[P5]] [6] : i32 into vector<8xi32>
        %t = xegpu.create_nd_tdesc %src : memref<?x?x?x?xf16> -> !xegpu.tensor_desc<1x1x8x16xf16>
        %c = builtin.unrealized_conversion_cast %t : !xegpu.tensor_desc<1x1x8x16xf16> to vector<8xi32>
        gpu.return %c : vector<8xi32>
    }

    // Batched (>2D) with the maximum three leading dims, over a subview of a
    // wider source taken at a nonzero offset. The innermost planes are 64 rows
    // apart but only 32 rows tall, so the planes are not packed back to back.
    // The subview's offset belongs to base_ptr, so base_height stays the row
    // extent measured from that shifted base:
    //   32 + 1 * 768 + 1 * 256 + 3 * 64 = 1248
    // and the furthest row the batch dims reach is 768 + 256 + 3 * 64 + 31 =
    // 1247, just inside it. All three spare payload slots carry a batch row
    // stride.
    // CHECK-LABEL: gpu.func @create_nd_tdesc_batch_subview(
    // CHECK-SAME:  %[[SRC:.+]]: memref<3x3x4x64x64xf32>
    gpu.func @create_nd_tdesc_batch_subview(%src: memref<3x3x4x64x64xf32>) -> vector<8xi32> {
        // The subview offset 1 * 49152 + 1 * 16384 + 16 * 64 = 66560 folds into
        // base_ptr and leaves the shape fields alone.
        // CHECK: %[[INTPTR:.+]] = memref.extract_aligned_pointer_as_index %{{.+}}
        // CHECK: %[[OFF:.+]] = arith.constant 66560 : index
        // CHECK: %[[PTR_I64:.+]] = arith.index_castui %[[INTPTR]] : index to i64
        // CHECK: %[[OFF_I64:.+]] = arith.index_castui %[[OFF]] : index to i64
        // CHECK: %[[ELEM_SZ:.+]] = arith.constant 4 : i64
        // CHECK: %[[OFF_BYTES:.+]] = arith.muli %[[OFF_I64]], %[[ELEM_SZ]] : i64
        // CHECK: %[[BASE_PTR:.+]] = arith.addi %[[PTR_I64]], %[[OFF_BYTES]] : i64
        // W = size[4] = 32, pitch = stride[3] = 64.
        // CHECK: %[[W:.+]] = arith.trunci %{{.+}} : i64 to i32
        // CHECK: %[[PITCH:.+]] = arith.trunci %{{.+}} : i64 to i32
        // Batch row strides = stride[d] / pitch: 49152 / 64, 16384 / 64, 4096 / 64.
        // CHECK: %[[ROWS0:.+]] = arith.constant 768 : i32
        // CHECK: %[[ROWS1:.+]] = arith.constant 256 : i32
        // CHECK: %[[ROWS2:.+]] = arith.constant 64 : i32
        // base_height = size[3] + sum_d (size[d] - 1) * rows_d.
        // CHECK: %[[H:.+]] = arith.trunci %{{.+}} : i64 to i32
        // CHECK: %[[C1I32:.+]] = arith.constant 1 : i32
        // CHECK: %[[BATCH0:.+]] = arith.trunci %{{.+}} : i64 to i32
        // CHECK: %[[BATCH0M1:.+]] = arith.subi %[[BATCH0]], %[[C1I32]] : i32
        // CHECK: %[[BATCH0_ROWS:.+]] = arith.muli %[[BATCH0M1]], %[[ROWS0]] : i32
        // CHECK: %[[FLAT_H0:.+]] = arith.addi %[[H]], %[[BATCH0_ROWS]] : i32
        // CHECK: %[[BATCH1:.+]] = arith.trunci %{{.+}} : i64 to i32
        // CHECK: %[[BATCH1M1:.+]] = arith.subi %[[BATCH1]], %[[C1I32]] : i32
        // CHECK: %[[BATCH1_ROWS:.+]] = arith.muli %[[BATCH1M1]], %[[ROWS1]] : i32
        // CHECK: %[[FLAT_H1:.+]] = arith.addi %[[FLAT_H0]], %[[BATCH1_ROWS]] : i32
        // CHECK: %[[BATCH2:.+]] = arith.trunci %{{.+}} : i64 to i32
        // CHECK: %[[BATCH2M1:.+]] = arith.subi %[[BATCH2]], %[[C1I32]] : i32
        // CHECK: %[[BATCH2_ROWS:.+]] = arith.muli %[[BATCH2M1]], %[[ROWS2]] : i32
        // CHECK: %[[FLAT_H:.+]] = arith.addi %[[FLAT_H1]], %[[BATCH2_ROWS]] : i32
        // CHECK: %[[PI64:.+]] = vector.insert %[[BASE_PTR]], %{{.+}} [0] : i64 into vector<4xi64>
        // CHECK: %[[P2:.+]] = vector.insert %[[W]], %{{.+}} [2] : i32 into vector<8xi32>
        // CHECK: %[[P3:.+]] = vector.insert %[[FLAT_H]], %[[P2]] [3] : i32 into vector<8xi32>
        // CHECK: %[[P4:.+]] = vector.insert %[[PITCH]], %[[P3]] [4] : i32 into vector<8xi32>
        // CHECK: %[[P5:.+]] = vector.insert %[[ROWS0]], %[[P4]] [5] : i32 into vector<8xi32>
        // CHECK: %[[P6:.+]] = vector.insert %[[ROWS1]], %[[P5]] [6] : i32 into vector<8xi32>
        // CHECK: vector.insert %[[ROWS2]], %[[P6]] [7] : i32 into vector<8xi32>
        %sub = memref.subview %src[1, 1, 0, 16, 0] [2, 2, 4, 32, 32] [1, 1, 1, 1, 1]
            : memref<3x3x4x64x64xf32>
           to memref<2x2x4x32x32xf32, strided<[49152, 16384, 4096, 64, 1], offset: 66560>>
        %t = xegpu.create_nd_tdesc %sub
            : memref<2x2x4x32x32xf32, strided<[49152, 16384, 4096, 64, 1], offset: 66560>>
            -> !xegpu.tensor_desc<1x1x1x8x16xf32>
        %c = builtin.unrealized_conversion_cast %t : !xegpu.tensor_desc<1x1x1x8x16xf32> to vector<8xi32>
        gpu.return %c : vector<8xi32>
    }
}

// -----
// A row stride of 17 f32 elements is a 68 byte pitch. Xe3p only requires the
// pitch to be 4 byte aligned, so this lowers (Xe2 requires 16 and rejects it,
// see failed_conversion.mlir).
gpu.module @create_nd_tdesc_pitch_xe3p [#xevm.target<chip = "cri">] {
    // CHECK-LABEL: gpu.func @pitch_4byte_aligned
    gpu.func @pitch_4byte_aligned(%src: memref<8x16xf32, strided<[17, 1]>>) -> vector<8xi32> {
        // CHECK: %[[C17:.*]] = arith.constant 17 : i64
        // CHECK: %[[PITCH:.*]] = arith.trunci %[[C17]] : i64 to i32
        // CHECK: vector.insert %[[PITCH]], %{{.*}} [4] : i32 into vector<8xi32>
        %t = xegpu.create_nd_tdesc %src : memref<8x16xf32, strided<[17, 1]>> -> !xegpu.tensor_desc<8x16xf32>
        %c = builtin.unrealized_conversion_cast %t : !xegpu.tensor_desc<8x16xf32> to vector<8xi32>
        gpu.return %c : vector<8xi32>
    }
}

// -----
// Xe3p has no minimum width or pitch, only a 4 byte granularity, so a 16 byte
// wide f16 surface lowers. Xe2 requires 32 bytes and rejects the same source,
// see failed_conversion.mlir.
gpu.module @create_nd_tdesc_narrow_xe3p [#xevm.target<chip = "cri">] {
    // CHECK-LABEL: gpu.func @narrow_surface_xe3p
    gpu.func @narrow_surface_xe3p(%src: memref<8x8xf16, strided<[16, 1]>>) -> vector<8xi32> {
        // CHECK: vector.insert %{{.*}} [4] : i32 into vector<8xi32>
        %t = xegpu.create_nd_tdesc %src : memref<8x8xf16, strided<[16, 1]>> -> !xegpu.tensor_desc<8x8xf16>
        %c = builtin.unrealized_conversion_cast %t : !xegpu.tensor_desc<8x8xf16> to vector<8xi32>
        gpu.return %c : vector<8xi32>
    }
}

// -----
// A memref with dynamic shape or strides carries nothing statically checkable,
// and its shape/strides are recovered via memref.extract_strided_metadata
// rather than from the op, so the restriction check must skip it entirely
// instead of querying the op's mixed sizes.
gpu.module @create_nd_tdesc_dyn_chip [#xevm.target<chip = "pvc">] {
    // CHECK-LABEL: gpu.func @dynamic_source_with_chip
    gpu.func @dynamic_source_with_chip(%src: memref<?x?xf16>) -> vector<8xi32> {
        // CHECK: memref.extract_strided_metadata
        // CHECK: vector.insert %{{.*}} [4] : i32 into vector<8xi32>
        %t = xegpu.create_nd_tdesc %src : memref<?x?xf16> -> !xegpu.tensor_desc<8x16xf16>
        %c = builtin.unrealized_conversion_cast %t : !xegpu.tensor_desc<8x16xf16> to vector<8xi32>
        gpu.return %c : vector<8xi32>
    }
}

// -----
// Without a target chip there is no uArch to check against, so a surface too
// narrow for Xe2 is still lowered as-is.
gpu.module @create_nd_tdesc_no_chip {
    // CHECK-LABEL: gpu.func @narrow_surface_without_chip
    gpu.func @narrow_surface_without_chip(%src: memref<8x8xf16, strided<[16, 1]>>) -> vector<8xi32> {
        // CHECK: vector.insert %{{.*}} [4] : i32 into vector<8xi32>
        %t = xegpu.create_nd_tdesc %src : memref<8x8xf16, strided<[16, 1]>> -> !xegpu.tensor_desc<8x8xf16>
        %c = builtin.unrealized_conversion_cast %t : !xegpu.tensor_desc<8x8xf16> to vector<8xi32>
        gpu.return %c : vector<8xi32>
    }
}
