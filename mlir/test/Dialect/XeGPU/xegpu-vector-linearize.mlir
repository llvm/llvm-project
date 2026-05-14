// RUN: mlir-opt %s -split-input-file -xegpu-vector-linearize -canonicalize | FileCheck %s

// CHECK-LABEL: test_vector_insert_2d_idx
// CHECK-SAME: (%[[DEST:.*]]: vector<2x8x4xf32>, %[[SRC:.*]]: vector<4xf32>) -> vector<2x8x4xf32>
// CHECK: %[[ARG_DEST:.*]] = vector.shape_cast %[[DEST]] : vector<2x8x4xf32> to vector<64xf32>
// CHECK: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG_DEST]], %[[SRC]]
// CHECK: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 64, 65, 66, 67, 16, 17, 18, 19, 20, 21,
// CHECK-SAME: 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
// CHECK-SAME: 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<64xf32>, vector<4xf32>
// CHECK: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<64xf32> to vector<2x8x4xf32>
// CHECK: return %[[RES]] : vector<2x8x4xf32>
func.func @test_vector_insert_2d_idx(%arg0: vector<2x8x4xf32>, %arg1: vector<4xf32>) -> vector<2x8x4xf32> {
  %0 = vector.insert %arg1, %arg0[0, 3]: vector<4xf32> into vector<2x8x4xf32>
  return %0 : vector<2x8x4xf32>
}

// -----
// CHECK-LABEL: test_vector_transpose
// CHECK-SAME: (%[[ORIG_ARG:.*]]: vector<2x8xf32>) -> vector<8x2xf32>
// CHECK: %[[POISON:.*]] = ub.poison : vector<16xf32>
// CHECK: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<2x8xf32> to vector<16xf32>
// CHECK: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG]], %[[POISON]]
// CHECK: [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<16xf32>, vector<16xf32>
// CHECK: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<16xf32> to vector<8x2xf32>
// CHECK: return %[[RES]] : vector<8x2xf32>
func.func @test_vector_transpose(%arg: vector<2x8xf32>) -> vector<8x2xf32> {
  %0 = vector.transpose %arg, [1, 0] : vector<2x8xf32> to vector<8x2xf32>
  return %0 : vector<8x2xf32>
}

// -----
// CHECK-LABEL: test_vector_transpose_16x16
// CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
// CHECK-62: vector.shuffle
func.func @test_vector_transpose_16x16(%arg: vector<16x16xf32>) -> vector<16x16xf32> {
  %0 = vector.transpose %arg, [1, 0] : vector<16x16xf32> to vector<16x16xf32>
  return %0 : vector<16x16xf32>
}

// -----

// CHECK-LABEL: func.func @test_vector_store_load_4x4_f16
// CHECK-SAME: (%[[ARG0:.*]]: memref<4x4xf16>)
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[LOAD0:.*]] = vector.load %[[ARG0]][%[[C0]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
// CHECK: %[[LOAD1:.*]] = vector.load %[[ARG0]][%[[C1]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
// CHECK: %[[LOAD2:.*]] = vector.load %[[ARG0]][%[[C2]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
// CHECK: %[[LOAD3:.*]] = vector.load %[[ARG0]][%[[C3]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
// CHECK: vector.store %[[LOAD0]], %[[ARG0]][%[[C0]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
// CHECK: vector.store %[[LOAD1]], %[[ARG0]][%[[C1]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
// CHECK: vector.store %[[LOAD2]], %[[ARG0]][%[[C2]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
// CHECK: vector.store %[[LOAD3]], %[[ARG0]][%[[C3]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
func.func @test_vector_store_load_4x4_f16(%buffer: memref<4x4xf16>) {
  %c0 = arith.constant 0 : index
  %0 = vector.load %buffer[%c0, %c0] : memref<4x4xf16>, vector<4x4xf16>
  vector.store %0, %buffer[%c0, %c0] : memref<4x4xf16>, vector<4x4xf16>
  return
}

// -----
// CHECK-LABEL: func.func @test_vector_store_load_4x4x4
// CHECK-SAME: (%[[BUF:.*]]: memref<4x4x4xf32>)
// Constants (order not important)
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// All 16 scalar-slice (row/col plane) loads of 1D vectors
// CHECK-COUNT-16: vector.load {{.*}} : memref<4x4x4xf32>, vector<4xf32>
// No remaining 3D vector load
// CHECK-NOT: vector.load {{.*}} : memref<4x4x4xf32>, vector<4x4x4xf32>
// All 16 stores of 1D vectors
// CHECK-COUNT-16: vector.store {{.*}} : memref<4x4x4xf32>, vector<4xf32>
// CHECK: return
func.func @test_vector_store_load_4x4x4(%buffer: memref<4x4x4xf32>) {
  %c0 = arith.constant 0 : index
  %0 = vector.load %buffer[%c0, %c0, %c0] : memref<4x4x4xf32>, vector<4x4x4xf32>
  vector.store %0, %buffer[%c0, %c0, %c0] : memref<4x4x4xf32>, vector<4x4x4xf32>
  return
}

// -----
// CHECK-LABEL: func.func @test_linearize_index
// CHECK-SAME: (%[[ARG0:.*]]: vector<2x2xindex>, %[[ARG1:.*]]: vector<2x2xi32>) -> vector<2x2xindex>
// CHECK: %[[CST:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
// CHECK: %[[CAST1:.*]] = vector.shape_cast %[[ARG1]] : vector<2x2xi32> to vector<4xi32>
// CHECK: %[[CAST2:.*]] = vector.shape_cast %[[ARG0]] : vector<2x2xindex> to vector<4xindex>
// CHECK: %[[ADDI:.*]] = arith.addi %[[CAST2]], %[[CST]] : vector<4xindex>
// CHECK: %[[INDEX_CAST1:.*]] = arith.index_cast %[[ADDI]] : vector<4xindex> to vector<4xi32>
// CHECK: %[[MULI:.*]] = arith.muli %[[INDEX_CAST1]], %[[CAST1]] : vector<4xi32>
// CHECK: %[[INDEX_CAST2:.*]] = arith.index_cast %[[MULI]] : vector<4xi32> to vector<4xindex>
// CHECK: %[[RESULT:.*]] = vector.shape_cast %[[INDEX_CAST2]] : vector<4xindex> to vector<2x2xindex>
// CHECK: return %[[RESULT]] : vector<2x2xindex>
func.func @test_linearize_index(%arg0: vector<2x2xindex>, %arg1: vector<2x2xi32>) -> vector<2x2xindex> {
  %0 = arith.constant dense<[[0, 1], [2, 3]]> : vector<2x2xindex>
  // Arith and math ops are handled in generic way, check some of them
  %1 = arith.addi %arg0, %0 :  vector<2x2xindex>
  %2 = arith.index_cast %1 : vector<2x2xindex> to vector<2x2xi32>
  %3 = arith.muli %2, %arg1 : vector<2x2xi32>
  %4 = arith.index_cast %3 : vector<2x2xi32> to vector<2x2xindex>
  return %4 : vector<2x2xindex>
}

// -----
// CHECK-LABEL: func.func @broadcast_stretch_at_start
// CHECK-SAME: (%[[ARG0:.*]]: vector<1x4xf32>) -> vector<3x4xf32>
// CHECK: %[[POISON:.*]] = ub.poison : vector<12xf32>
// CHECK: %[[CAST:.*]] = vector.shape_cast %[[ARG0]] : vector<1x4xf32> to vector<4xf32>
// CHECK: %[[SHUFFLE1:.*]] = vector.shuffle %[[POISON]], %[[CAST]] [12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1] : vector<12xf32>, vector<4xf32>
// CHECK: %[[SHUFFLE2:.*]] = vector.shuffle %[[SHUFFLE1]], %[[CAST]] [0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11] : vector<12xf32>, vector<4xf32>
// CHECK: %[[SHUFFLE3:.*]] = vector.shuffle %[[SHUFFLE2]], %[[CAST]] [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15] : vector<12xf32>, vector<4xf32>
// CHECK: %[[RESULT:.*]] = vector.shape_cast %[[SHUFFLE3]] : vector<12xf32> to vector<3x4xf32>
func.func @broadcast_stretch_at_start(%arg0: vector<1x4xf32>) -> vector<3x4xf32> {
  %0 = vector.broadcast %arg0 : vector<1x4xf32> to vector<3x4xf32>
  return %0 : vector<3x4xf32>
}

// -----
// CHECK-LABEL: func.func @broadcast_stretch_at_end
// CHECK-SAME: (%[[ARG0:.*]]: vector<4x1xf32>) -> vector<4x3xf32>
// CHECK: %[[POISON:.*]] = ub.poison : vector<12xf32>
// CHECK: %[[EXTRACT1:.*]] = vector.extract %[[ARG0]][0, 0] : f32 from vector<4x1xf32>
// CHECK: %[[BROADCAST1:.*]] = vector.broadcast %[[EXTRACT1]] : f32 to vector<3xf32>
// CHECK: vector.shuffle
// CHECK: %[[EXTRACT2:.*]] = vector.extract %[[ARG0]][1, 0] : f32 from vector<4x1xf32>
// CHECK: %[[BROADCAST2:.*]] = vector.broadcast %[[EXTRACT2]] : f32 to vector<3xf32>
// CHECK: vector.shuffle
// CHECK: %[[EXTRACT3:.*]] = vector.extract %[[ARG0]][2, 0] : f32 from vector<4x1xf32>
// CHECK: %[[BROADCAST3:.*]] = vector.broadcast %[[EXTRACT3]] : f32 to vector<3xf32>
// CHECK: vector.shuffle
// CHECK: %[[EXTRACT4:.*]] = vector.extract %[[ARG0]][3, 0] : f32 from vector<4x1xf32>
// CHECK: %[[BROADCAST4:.*]] = vector.broadcast %[[EXTRACT4]] : f32 to vector<3xf32>
// CHECK: vector.shuffle
// CHECK: vector.shape_cast {{.*}} : vector<12xf32> to vector<4x3xf32>
func.func @broadcast_stretch_at_end(%arg0: vector<4x1xf32>) -> vector<4x3xf32> {
  %0 = vector.broadcast %arg0 : vector<4x1xf32> to vector<4x3xf32>
  return %0 : vector<4x3xf32>
}

// -----
// CHECK-LABEL: func.func @broadcast_stretch_in_middle
// CHECK-SAME: (%[[ARG0:.*]]: vector<4x1x2xf32>) -> vector<4x3x2xf32>
// CHECK: ub.poison : vector<6xf32>
// CHECK: ub.poison : vector<24xf32>
// CHECK: %[[CAST:.*]] = vector.shape_cast %[[ARG0]] : vector<4x1x2xf32> to vector<8xf32>
// CHECK-COUNT-20: vector.shuffle
// CHECK: vector.shape_cast {{.*}} : vector<24xf32> to vector<4x3x2xf32>
// CHECK-NOT: vector.broadcast
func.func @broadcast_stretch_in_middle(%arg0: vector<4x1x2xf32>) -> vector<4x3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<4x1x2xf32> to vector<4x3x2xf32>
  return %0 : vector<4x3x2xf32>
}

// CHECK-LABEL: func.func @gather_memref_2d
// CHECK-SAME: (%[[BASE:.*]]: memref<?x?xf32>, %[[IDX:.*]]: vector<2x3xindex>, %[[MASK:.*]]: vector<2x3xi1>, %[[PASS:.*]]: vector<2x3xf32>) -> vector<2x3xf32>

// CHECK: %[[POISON:.*]] = ub.poison : vector<6xf32>
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[PASS_CAST:.*]] = vector.shape_cast %[[PASS]] : vector<2x3xf32> to vector<6xf32>

// First shuffle + if ladder for row 0
// CHECK: %[[ROW0_INIT:.*]] = vector.shuffle %[[PASS_CAST]], %[[POISON]] [0, 1, 2]
// CHECK: %[[DIM0:.*]] = memref.dim %[[BASE]], %[[C0]]
// CHECK: %[[DIM1:.*]] = memref.dim %[[BASE]], %[[C1]]
// CHECK: %[[MASK_0_0:.*]] = vector.extract %[[MASK]][0, 0]
// CHECK: %[[IDX_0_0:.*]] = vector.extract %[[IDX]][0, 0]
// CHECK: %[[OFF_0_0:.*]] = arith.addi %[[IDX_0_0]], %[[C1]]
// CHECK: %[[DL_0_0:.*]]:2 = affine.delinearize_index %[[OFF_0_0]] into (%[[DIM0]], %[[DIM1]])
// CHECK: %[[IF_0_0:.*]] = scf.if %[[MASK_0_0]] -> (vector<3xf32>) {
// CHECK:   %[[LOAD_0_0:.*]] = vector.load %[[BASE]][%[[DL_0_0]]#0, %[[DL_0_0]]#1] : memref<?x?xf32>, vector<1xf32>
// CHECK:   %[[ELEM_0_0:.*]] = vector.extract %[[LOAD_0_0]][0] : f32
// CHECK:   %[[INS_0_0:.*]] = vector.insert %[[ELEM_0_0]], %[[ROW0_INIT]] [0] : f32 into vector<3xf32>
// CHECK:   scf.yield %[[INS_0_0]] : vector<3xf32>
// CHECK: } else {
// CHECK:   scf.yield %[[ROW0_INIT]] : vector<3xf32>
// CHECK: }

// CHECK: %[[MASK_0_1:.*]] = vector.extract %[[MASK]][0, 1]
// CHECK: %[[IDX_0_1:.*]] = vector.extract %[[IDX]][0, 1]
// CHECK: %[[OFF_0_1:.*]] = arith.addi %[[IDX_0_1]], %[[C1]]
// CHECK: %[[DL_0_1:.*]]:2 = affine.delinearize_index %[[OFF_0_1]] into (%[[DIM0]], %[[DIM1]])
// CHECK: %[[IF_0_1:.*]] = scf.if %[[MASK_0_1]] -> (vector<3xf32>)

// … (similar checks for the rest of row 0, then row 1)

// CHECK: %[[ROW_SHUFFLE:.*]] = vector.shuffle %[[POISON]], {{.*}} [6, 7, 8, -1, -1, -1]
// CHECK: %[[ROW1_INIT:.*]] = vector.shuffle %[[PASS_CAST]], %[[POISON]] [3, 4, 5]

// Row 1 if ladder checks
// CHECK: %[[MASK_1_0:.*]] = vector.extract %[[MASK]][1, 0]
// CHECK: %[[IDX_1_0:.*]] = vector.extract %[[IDX]][1, 0]
// CHECK: %[[OFF_1_0:.*]] = arith.addi %[[IDX_1_0]], %[[C1]]
// CHECK: %[[DL_1_0:.*]]:2 = affine.delinearize_index %[[OFF_1_0]] into
// CHECK: %[[IF_1_0:.*]] = scf.if %[[MASK_1_0]] -> (vector<3xf32>)

// … (similar checks for remaining row 1 inserts)

// Final reshuffle and cast
// CHECK: %[[FINAL_SHUFFLE:.*]] = vector.shuffle %[[ROW_SHUFFLE]], {{.*}} [0, 1, 2, 6, 7, 8]
// CHECK: %[[RESULT:.*]] = vector.shape_cast %[[FINAL_SHUFFLE]] : vector<6xf32> to vector<2x3xf32>
// CHECK: return %[[RESULT]] : vector<2x3xf32>
func.func @gather_memref_2d(%base: memref<?x?xf32>, %v: vector<2x3xindex>, %mask: vector<2x3xi1>, %pass_thru: vector<2x3xf32>) -> vector<2x3xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = vector.gather %base[%c0, %c1][%v], %mask, %pass_thru : memref<?x?xf32>, vector<2x3xindex>, vector<2x3xi1>, vector<2x3xf32> into vector<2x3xf32>
  return %0 : vector<2x3xf32>
}

// -----
// Check for vector linearization interoperability with XeGPU dialect ops.
// The `xegpu-vector-linearize` pass does not itself affect the XeGPU ops.

// CHECK: gpu.func @test_kernel(%[[A:.*]]: memref<8x16xf16>, %[[B:.*]]: memref<16x16xf16>, %[[C:.*]]: memref<8x16xf32>) kernel {
// CHECK: %[[POISON_F32:.*]] = ub.poison : vector<128xf32>
// CHECK: %[[CST_A:.*]] = arith.constant dense<0.000000e+00> : vector<64xf16>
// CHECK: %[[CST_C:.*]] = arith.constant dense<5.000000e+00> : vector<64xf32>

// CHECK: %[[A_TDESC:.*]] = xegpu.create_nd_tdesc %[[A]]
// CHECK: %[[A_VAL:.*]] = xegpu.load_nd %[[A_TDESC]][0, 0]
// CHECK: %[[A_CAST:.*]] = vector.shape_cast %[[A_VAL]] : vector<8x16xf16> to vector<128xf16>
// CHECK: %[[A_SHUFFLE:.*]] = vector.shuffle %[[A_CAST]], %[[CST_A]] {{.*}} : vector<128xf16>, vector<64xf16>
// CHECK: %[[A_RESULT:.*]] = vector.shape_cast %[[A_SHUFFLE]] : vector<128xf16> to vector<8x16xf16>

// CHECK: %[[B_TDESC:.*]] = xegpu.create_nd_tdesc %[[B]]
// CHECK: %[[B_VAL:.*]] = xegpu.load_nd %[[B_TDESC]][0, 0]
// CHECK: %[[B_CAST:.*]] = vector.shape_cast %[[B_VAL]] : vector<16x16xf16> to vector<256xf16>
// CHECK: %[[B_SHUFFLE:.*]] = vector.shuffle %[[B_CAST]], %[[CST_A]] {{.*}} : vector<256xf16>, vector<64xf16>
// CHECK: %[[B_RESULT:.*]] = vector.shape_cast %[[B_SHUFFLE]] : vector<256xf16> to vector<16x16xf16>

// CHECK: %[[DPAS:.*]] = xegpu.dpas %[[A_RESULT]], %[[B_RESULT]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
// CHECK: %[[DPAS_CAST:.*]] = vector.shape_cast %[[DPAS]] : vector<8x16xf32> to vector<128xf32>
// CHECK: %[[EXTRACT_SHUFFLE:.*]] = vector.shuffle %[[DPAS_CAST]], %[[POISON_F32]] {{.*}} : vector<128xf32>, vector<128xf32>
// CHECK: %[[ADDF:.*]] = arith.addf %[[EXTRACT_SHUFFLE]], %[[CST_C]] : vector<64xf32>
// CHECK: %[[INSERT_SHUFFLE:.*]] = vector.shuffle %[[DPAS_CAST]], %[[ADDF]] {{.*}} : vector<128xf32>, vector<64xf32>
// CHECK: %[[C_RESULT:.*]] = vector.shape_cast %[[INSERT_SHUFFLE]] : vector<128xf32> to vector<8x16xf32>

// CHECK: %[[C_TDESC:.*]] = xegpu.create_nd_tdesc %[[C]]
// CHECK: xegpu.store_nd %[[C_RESULT]], %[[C_TDESC]][0, 0]
// CHECK: gpu.return

gpu.module @test_kernel {
  gpu.func @test_kernel(%A: memref<8x16xf16>, %B: memref<16x16xf16>, %C: memref<8x16xf32>) kernel  {
    %c0 = arith.constant 0 : index
    %cst_vec_0 = arith.constant dense<0.000000e+00> : vector<8x8xf16>
    %cst_vec_1 = arith.constant dense<0.000000e+00> : vector<8x8xf16>
    %cst_vec_2 = arith.constant dense<5.000000e+00> : vector<8x8xf32>
    %a_tdesc = xegpu.create_nd_tdesc %A : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<array_length = 1>>
    %a_val = xegpu.load_nd %a_tdesc[0, 0] : !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<array_length = 1>> -> vector<8x16xf16>
    %a_val_0 = vector.insert_strided_slice %cst_vec_0, %a_val{offsets = [0, 0], sizes = [8, 8], strides = [1, 1]}: vector<8x8xf16> into vector<8x16xf16>
    %b_tdesc = xegpu.create_nd_tdesc %B : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1>>

    %b_val = xegpu.load_nd  %b_tdesc[0, 0] : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1>> -> vector<16x16xf16>
    %b_val_0 = vector.insert_strided_slice %cst_vec_1, %b_val{offsets = [0, 0], sizes = [8, 8], strides = [1, 1]}: vector<8x8xf16> into vector<16x16xf16>
    %c_val = xegpu.dpas %a_val_0, %b_val_0 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    %c_val_0 = vector.extract_strided_slice %c_val {offsets = [0, 0], sizes = [8, 8], strides = [1, 1]} : vector<8x16xf32> to vector<8x8xf32>
    %c_addf = arith.addf %c_val_0, %cst_vec_2 : vector<8x8xf32>
    %c_result = vector.insert_strided_slice %c_addf, %c_val {offsets = [0, 0], sizes = [8, 8], strides = [1, 1]} : vector<8x8xf32> into vector<8x16xf32>
    %c_tdesc = xegpu.create_nd_tdesc %C : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1>>
    xegpu.store_nd %c_result, %c_tdesc[0, 0] : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    gpu.return
  }
}


