// RUN: mlir-opt %s -split-input-file -pass-pipeline="func.func(convert-vector-to-gpu{use-nvgpu=true})" | FileCheck %s

//#########################################################
// INT8 row-row-row
//#########################################################

// CHECK-DAG: [[$rowA0_map:#.+]] = affine_map<()[s0] -> (s0 mod 16 + 1)>
// CHECK-DAG: [[$colA0_map:#.+]] = affine_map<()[s0] -> ((s0 floordiv 16) * 16 + 1)>

// CHECK-DAG: [[$rowB0_map:#.+]] = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 4) * 16 + 39)>
// CHECK-DAG: [[$colB0_map:#.+]] = affine_map<()[s0] -> (s0 floordiv 4 + 40)>
// CHECK-DAG: [[$rowB1_map:#.+]] = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 4) * 16 + 40)>
// CHECK-DAG: [[$rowB2_map:#.+]] = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 4) * 16 + 41)>
// CHECK-DAG: [[$rowB3_map:#.+]] = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 4) * 16 + 42)>
// CHECK-DAG: [[$rowB4_map:#.+]] = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 4) * 16 + 55)>
// CHECK-DAG: [[$rowB5_map:#.+]] = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 4) * 16 + 56)>
// CHECK-DAG: [[$rowB6_map:#.+]] = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 4) * 16 + 57)>
// CHECK-DAG: [[$rowB7_map:#.+]] = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 4) * 16 + 58)>

// CHECK-DAG: [[$rowC0_map:#.+]] = affine_map<()[s0] -> (s0 floordiv 4 + 49)>
// CHECK-DAG: [[$colC0_map:#.+]] = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 4) * 8 + 40)>
// CHECK-DAG: [[$rowC8_map:#.+]] = affine_map<()[s0] -> (s0 floordiv 4 + 57)>


#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @m16n8k32_int8_row_row_row
func.func @m16n8k32_int8_row_row_row(%arg0: memref<128x128xi8, 3>, %arg1: memref<128x128xi8, 3>, %arg2: memref<128x128xi32>) {
  %cst_0 = arith.constant dense<0> : vector<32x8xi8>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c17 = arith.constant 17 : index  
  %c39 = arith.constant 39 : index  
  %c40 = arith.constant 40 : index  
  %c49 = arith.constant 49 : index  
  %c50 = arith.constant 50 : index  
  %cst = arith.constant 0 : i8
  %cst0 = arith.constant 0 : i32

  // Verify that the operand A is distributed to loads correctly.

  // CHECK: [[row:%.+]] = affine.apply [[$rowA0_map]]()[{{%.+}}]
  // CHECK: [[col:%.+]] = affine.apply [[$colA0_map]]()[{{%.+}}]
  // CHECK: nvgpu.ldmatrix %arg0[[[row]], [[col]]] {numTiles = 4 : i32, transpose = false} : memref<128x128xi8, 3> -> vector<4x4xi8>

  // Verify that the operand B is distributed to loads correctly. It's elements
  // must be loaded in a non-vectorized manner to do the transpose.

  // CHECK-DAG: [[row:%.+]] = affine.apply [[$rowB0_map]]()[{{%.+}}]
  // CHECK-DAG: [[col:%.+]] = affine.apply [[$colB0_map]]()[{{%.+}}]
  // CHECK: memref.load %arg1[[[row]], [[col]]] : memref<128x128xi8, 3>
  // CHECK-DAG: [[row:%.+]] = affine.apply [[$rowB1_map]]()[{{%.+}}]
  // CHECK-DAG: [[col:%.+]] = affine.apply [[$colB0_map]]()[{{%.+}}]
  // CHECK: memref.load %arg1[[[row]], [[col]]] : memref<128x128xi8, 3>
  // CHECK-DAG: [[row:%.+]] = affine.apply [[$rowB2_map]]()[{{%.+}}]
  // CHECK-DAG: [[col:%.+]] = affine.apply [[$colB0_map]]()[{{%.+}}]
  // CHECK: memref.load %arg1[[[row]], [[col]]] : memref<128x128xi8, 3>
  // CHECK-DAG: [[row:%.+]] = affine.apply [[$rowB3_map]]()[{{%.+}}]
  // CHECK-DAG: [[col:%.+]] = affine.apply [[$colB0_map]]()[{{%.+}}]
  // CHECK: memref.load %arg1[[[row]], [[col]]] : memref<128x128xi8, 3>
  // CHECK-DAG: [[col:%.+]] = affine.apply [[$colB0_map]]()[{{%.+}}]
  // CHECK-DAG: [[row:%.+]] = affine.apply [[$rowB4_map]]()[{{%.+}}]  
  // CHECK: memref.load %arg1[[[row]], [[col]]] : memref<128x128xi8, 3>
  // CHECK-DAG: [[row:%.+]] = affine.apply [[$rowB5_map]]()[{{%.+}}]
  // CHECK-DAG: [[col:%.+]] = affine.apply [[$colB0_map]]()[{{%.+}}]
  // CHECK: memref.load %arg1[[[row]], [[col]]] : memref<128x128xi8, 3>
  // CHECK-DAG: [[row:%.+]] = affine.apply [[$rowB6_map]]()[{{%.+}}]
  // CHECK-DAG: [[col:%.+]] = affine.apply [[$colB0_map]]()[{{%.+}}]
  // CHECK: memref.load %arg1[[[row]], [[col]]] : memref<128x128xi8, 3>
  // CHECK-DAG: [[row:%.+]] = affine.apply [[$rowB7_map]]()[{{%.+}}]
  // CHECK-DAG: [[col:%.+]] = affine.apply [[$colB0_map]]()[{{%.+}}]
  // CHECK: memref.load %arg1[[[row]], [[col]]] : memref<128x128xi8, 3>
  // CHECK-NOT: memref.load %arg1

  // Verify that the operand C is distributed to loads correctly.
  // CHECK: [[row:%.+]] = affine.apply [[$rowC0_map]]()[{{%.+}}]
  // CHECK: [[col:%.+]] = affine.apply [[$colC0_map]]()[{{%.+}}]
  // CHECK: vector.load %arg2[[[row]], [[col]]] : memref<128x128xi32>, vector<2xi32>
  // CHECK: [[row:%.+]] = affine.apply [[$rowC8_map]]()[{{%.+}}]
  // CHECK: [[col:%.+]] = affine.apply [[$colC0_map]]()[{{%.+}}]
  // CHECK: vector.load %arg2[[[row]], [[col]]] : memref<128x128xi32>, vector<2xi32>
  // CHECK-NOT: vector.load %arg2{{.*}}

  %A = vector.transfer_read %arg0[%c1, %c1], %cst {in_bounds = [true, true]} : memref<128x128xi8, 3>, vector<16x32xi8>
  %B = vector.transfer_read %arg1[%c39, %c40], %cst {in_bounds = [true, true], permutation_map = #map0} : memref<128x128xi8, 3>, vector<8x32xi8>
  %C = vector.transfer_read %arg2[%c49, %c40], %cst0 {in_bounds = [true, true]} : memref<128x128xi32>, vector<16x8xi32>
  // CHECK: [[d:%.+]] = nvgpu.mma.sync({{.*}}) {mmaShape = [16, 8, 32]} : (vector<4x4xi8>, vector<2x4xi8>, vector<2x2xi32>) -> vector<2x2xi32>
  %D = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<16x32xi8>, vector<8x32xi8> into vector<16x8xi32>

  // CHECK: [[row:%.+]] = affine.apply [[$rowC0_map]]()[{{%.+}}]
  // CHECK: [[col:%.+]] = affine.apply [[$colC0_map]]()[{{%.+}}]
  // CHECK: vector.store {{%.+}}, %arg2[[[row]], [[col]]] : memref<128x128xi32>, vector<2xi32>
  // CHECK: [[row:%.+]] = affine.apply [[$rowC8_map]]()[{{%.+}}]
  // CHECK: [[col:%.+]] = affine.apply [[$colC0_map]]()[{{%.+}}]
  // CHECK: vector.store {{%.+}}, %arg2[[[row]], [[col]]] : memref<128x128xi32>, vector<2xi32>
  vector.transfer_write %D, %arg2[%c49, %c40] {in_bounds = [true, true]} : vector<16x8xi32>, memref<128x128xi32>
  return
}

// -----

//#########################################################
// f64 row-row-row
//#########################################################
// CHECK-DAG: [[$rowA0_map:#.+]] = affine_map<()[s0] -> (s0 floordiv 4 + 1)>
// CHECK-DAG: [[$colA0_map:#.+]] = affine_map<()[s0] -> (s0 mod 4 + 1)>

// CHECK-DAG: [[$rowb0_map:#.+]] = affine_map<()[s0] -> (s0 mod 4 + 39)>
// CHECK-DAG: [[$colb0_map:#.+]] = affine_map<()[s0] -> (s0 floordiv 4 + 40)>

// CHECK-DAG: [[$rowC0_map:#.+]] = affine_map<()[s0] -> (s0 floordiv 4 + 49)>
// CHECK-DAG: [[$colC0_map:#.+]] = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 4) * 8 + 40)

#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @m8n8k4_f64_row_row_row
func.func @m8n8k4_f64_row_row_row(%arg0: memref<128x128xf64>, %arg1: memref<128x128xf64>, %arg2: memref<128x128xf64>) {
  %cst_0 = arith.constant dense<0.0> : vector<4x8xf64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c17 = arith.constant 17 : index  
  %c39 = arith.constant 39 : index  
  %c40 = arith.constant 40 : index  
  %c49 = arith.constant 49 : index  
  %c50 = arith.constant 50 : index  
  %cst = arith.constant 0.0 : f64
  %cst0 = arith.constant 0.0 : f64

  // Verify that the operand A is distributed to loads correctly.

  // CHECK-DAG: [[row:%.+]] = affine.apply [[$rowA0_map]]
  // CHECK-DAG: [[col:%.+]] = affine.apply [[$colA0_map]]
  // CHECK: vector.load %arg0[[[row]], [[col]]] : memref<128x128xf64>, vector<1xf64>

  // Verify that the operand B is distributed to loads correctly. It's elements
  // must be loaded in a non-vectorized manner to do the transpose.

  // CHECK-DAG: [[row:%.+]] = affine.apply [[$rowb0_map]]
  // CHECK-DAG: [[col:%.+]] = affine.apply [[$colb0_map]]
  // CHECK: memref.load %arg1[[[row]], [[col]]] : memref<128x128xf64>

  // CHECK-DAG: [[row:%.+]] = affine.apply [[$rowC0_map]]
  // CHECK-DAG: [[col:%.+]] = affine.apply [[$colC0_map]]
  // CHECK: vector.load %arg2[[[row]], [[col]]] : memref<128x128xf64>, vector<2xf64>  

  %A = vector.transfer_read %arg0[%c1, %c1], %cst {in_bounds = [true, true]} : memref<128x128xf64>, vector<8x4xf64>
  %B = vector.transfer_read %arg1[%c39, %c40], %cst {in_bounds = [true, true], permutation_map = #map0} : memref<128x128xf64>, vector<8x4xf64>
  %C = vector.transfer_read %arg2[%c49, %c40], %cst0 {in_bounds = [true, true]} : memref<128x128xf64>, vector<8x8xf64>
  // CHECK: [[d:%.+]] = nvgpu.mma.sync({{.*}}) {mmaShape = [8, 8, 4]} : (vector<1x1xf64>, vector<1x1xf64>, vector<1x2xf64>) -> vector<1x2xf64>
  %D = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<8x4xf64>, vector<8x4xf64> into vector<8x8xf64>

  // CHECK-DAG: [[row:%.+]] = affine.apply [[$rowC0_map]]
  // CHECK-DAG: [[col:%.+]] = affine.apply [[$colC0_map]]
  // CHECK: vector.store {{%.+}}, %arg2[[[row]], [[col]]] : memref<128x128xf64>, vector<2xf64>  
  vector.transfer_write %D, %arg2[%c49, %c40] {in_bounds = [true, true]} : vector<8x8xf64>, memref<128x128xf64>
  return
}

// -----

//#########################################################
// FP16 row-row-row
//#########################################################

#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-DAG: [[$rowA_map:#.+]] = affine_map<()[s0] -> (s0 mod 16 + 1)>
// CHECK-DAG: [[$colA_map:#.+]] = affine_map<()[s0] -> ((s0 floordiv 16) * 8 + 3)>

// CHECK-DAG: [[$rowB_map:#.+]] = affine_map<()[s0] -> (s0 + 3)>
// CHECK-DAG: [[$colB_map:#.+]] = affine_map<() -> (3)>

// CHECK-LABEL: func @m16n8k16_fp16_row_row_row
func.func @m16n8k16_fp16_row_row_row(%arg0: memref<20x20xf16, 3>, %arg1: memref<20x20xf16, 3>, %arg2: memref<20x20xf16, 3>) {
  %cst_0 = arith.constant dense<0.000000e+00> : vector<16x8xf16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %cst = arith.constant 0.000000e+00 : f16
  // CHECK-DAG: [[row:%.+]] = affine.apply [[$rowA_map]]
  // CHECK-DAG: [[col:%.+]] = affine.apply [[$colA_map]]
  // CHECK: nvgpu.ldmatrix %arg0[[[row]], [[col]]] {numTiles = 4 : i32, transpose = false}  

  // CHECK-DAG: [[row:%.+]] = affine.apply [[$rowB_map]]
  // CHECK-DAG: [[col:%.+]] = affine.apply [[$colB_map]]
  // CHECK: nvgpu.ldmatrix %arg1[[[row]], [[col]]] {numTiles = 2 : i32, transpose = true}
  %A = vector.transfer_read %arg0[%c1, %c3], %cst {in_bounds = [true, true]} : memref<20x20xf16, 3>, vector<16x16xf16>
  %B = vector.transfer_read %arg1[%c3, %c3], %cst {permutation_map = #map0, in_bounds = [true, true]} : memref<20x20xf16, 3>, vector<8x16xf16>
  %C = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<20x20xf16, 3>, vector<16x8xf16>
  %D = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
  vector.transfer_write %D, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<20x20xf16, 3>
  return
}

// -----

// CHECK-DAG: [[$Arow_map:#.+]] = affine_map<()[s0] -> (s0 mod 16 + 1)>
// CHECK-DAG: [[$Acol_map:#.+]] = affine_map<()[s0] -> ((s0 floordiv 16) * 8 + 3)>
// CHECK-DAG: [[$Bcol_map:#.+]] = affine_map<() -> (3)>
// CHECK-DAG: [[$Brow_map:#.+]] = affine_map<()[s0] -> (s0 + 3)>

#map0 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @batch_m16n8k16_fp16_row_row_row
func.func @batch_m16n8k16_fp16_row_row_row(%arg0: memref<2x20x20xf16, 3>, %arg1: memref<2x20x20xf16, 3>, %arg2: memref<2x20x20xf16, 3>) {
  %cst_0 = arith.constant dense<0.000000e+00> : vector<20x20xf16>
  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %cst = arith.constant 0.000000e+00 : f16
  
  // CHECK-DAG: [[row:%.+]] = affine.apply [[$Arow_map]]
  // CHECK-DAG: [[col:%.+]] = affine.apply [[$Acol_map]]
  // CHECK: nvgpu.ldmatrix %arg0[[[C0]], [[row]], [[col]]] {numTiles = 4 : i32, transpose = false} : memref<2x20x20xf16, 3> -> vector<4x2xf16>
  %A = vector.transfer_read %arg0[%c0, %c1, %c3], %cst {in_bounds = [true, true]} : memref<2x20x20xf16, 3>, vector<16x16xf16>
  
  // CHECK-DAG: [[row:%.+]] = affine.apply [[$Brow_map]]
  // CHECK-DAG: [[col:%.+]] = affine.apply [[$Bcol_map]]  
  // CHECK: nvgpu.ldmatrix %arg1[[[C0]], [[row]], [[col]]] {numTiles = 2 : i32, transpose = true} : memref<2x20x20xf16, 3> -> vector<2x2xf16>
  %B = vector.transfer_read %arg1[%c0, %c3, %c3], %cst {permutation_map = #map0, in_bounds = [true, true]} : memref<2x20x20xf16, 3>, vector<8x16xf16>
  
  // CHECK-DAG: [[row:%.+]] = affine.apply [[$Arow_map]]
  // CHECK-DAG: [[col:%.+]] = affine.apply [[$Acol_map]]
  // CHECK: nvgpu.ldmatrix %arg2[[[C0]], [[row]], [[col]]] {numTiles = 2 : i32, transpose = false} : memref<2x20x20xf16, 3> -> vector<2x2xf16>
  %C = vector.transfer_read %arg2[%c0, %c1, %c3], %cst {in_bounds = [true, true]} : memref<2x20x20xf16, 3>, vector<16x8xf16>
  %D = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
  vector.transfer_write %D, %arg2[%c0, %c1, %c3] {in_bounds = [true, true]} : vector<16x8xf16>, memref<2x20x20xf16, 3>
  return
}

// -----

//#########################################################
// FP16 row-col-row
//#########################################################

#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK: [[$rowA_map:#.+]] = affine_map<()[s0] -> (s0 mod 16 + 1)>
// CHECK: [[$colA_map:#.+]] = affine_map<()[s0] -> ((s0 floordiv 16) * 8 + 3)>

// CHECK: [[$rowB_map:#.+]] = affine_map<()[s0] -> (s0 mod 8 + 1)>
// CHECK: [[$colB_map:#.+]] = affine_map<()[s0] -> ((s0 floordiv 8) * 8 + 3)>

// CHECK-LABEL: func @m16n8k16_fp16_row_col_row
func.func @m16n8k16_fp16_row_col_row(%arg0: memref<20x20xf16, 3>, %arg1: memref<20x20xf16, 3>, %arg2: memref<20x20xf16, 3>) {
  %cst_0 = arith.constant dense<0.000000e+00> : vector<16x8xf16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %cst = arith.constant 0.000000e+00 : f16
  // CHECK-DAG: [[row:%.+]] = affine.apply [[$rowA_map]]
  // CHECK-DAG: [[col:%.+]] = affine.apply [[$colA_map]]
  // CHECK: nvgpu.ldmatrix %arg0[[[row]], [[col]]] {numTiles = 4 : i32
  // CHECK-SAME: transpose = false
  
  // CHECK-DAG: [[row:%.+]] = affine.apply [[$rowB_map]]
  // CHECK-DAG: [[col:%.+]] = affine.apply [[$colB_map]]
  // CHECK: nvgpu.ldmatrix %arg1[[[row]], [[col]]] {numTiles = 2 : i32
  // CHECK-SAME: transpose = false

  // CHECK-DAG: [[row:%.+]] = affine.apply [[$rowA_map]]
  // CHECK-DAG: [[col:%.+]] = affine.apply [[$colA_map]]   
  // CHECK: nvgpu.ldmatrix %arg2[[[row]], [[col]]] {numTiles = 2 : i32
  // CHECK-SAME: transpose = false
  %A = vector.transfer_read %arg0[%c1, %c3], %cst {in_bounds = [true, true]} : memref<20x20xf16, 3>, vector<16x16xf16>
  %B = vector.transfer_read %arg1[%c1, %c3], %cst {in_bounds = [true, true]} : memref<20x20xf16, 3>, vector<8x16xf16>
  %C = vector.transfer_read %arg2[%c1, %c3], %cst {in_bounds = [true, true]} : memref<20x20xf16, 3>, vector<16x8xf16>
  %D = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
  vector.transfer_write %D, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<20x20xf16, 3>
  return
}

// -----

//#########################################################
// TF32 (multiplicand) F32 (accumulator) row-row-row
//#########################################################

#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-DAG: [[$rowA_map:#.+]] = affine_map<()[s0] -> (s0 mod 16 + 1)>
// CHECK-DAG: [[$colA_map:#.+]] = affine_map<()[s0] -> ((s0 floordiv 16) * 4 + 3)>

// CHECK-DAG: [[$rowB_map:#.+]] = affine_map<()[s0] -> (s0 mod 4 + 3)>
// CHECK-DAG: [[$colB_map:#.+]] = affine_map<()[s0] -> (s0 floordiv 4 + 3)>

// CHECK-DAG: [[$rowC_map:#.+]] = affine_map<()[s0] -> (s0 floordiv 4)>
// CHECK-DAG: [[$rowC8_map:#.+]] = affine_map<()[s0] -> (s0 floordiv 4 + 8)>
// CHECK-DAG: [[$colC_map:#.+]] = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 4) * 8)>

// CHECK-LABEL: func @m16n8k4_tf32_f32_row_row_row
func.func @m16n8k4_tf32_f32_row_row_row(%arg0: memref<20x20xf32, 3>, %arg1: memref<20x20xf32, 3>, %arg2: memref<20x20xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : vector<16x8xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %cst = arith.constant 0.000000e+00 : f32

  // CHECK: [[c_frag:%.+]] = arith.constant {{.*}} : vector<2x2xf32>

  // CHECK-DAG: [[row:%.+]] = affine.apply [[$rowA_map]]
  // CHECK-DAG: [[col:%.+]] = affine.apply [[$colA_map]]
  // CHECK: [[a_frag:%.+]] = nvgpu.ldmatrix %arg0[[[row]], [[col]]] {numTiles = 2 : i32, transpose = false}  

  // b and c are not loaded by ldmatrix in this test.
  // CHECK-NOT: nvgpu.ldmatrix

  // CHECK-DAG: [[row:%.+]] = affine.apply [[$rowB_map]]
  // CHECK-DAG: [[col:%.+]] = affine.apply [[$colB_map]]
  // CHECK: [[b_el:%.+]] = memref.load {{%.+}} : memref<20x20xf32, 3>  
  // CHECK: [[b_frag:%.+]] = vector.insert [[b_el]], {{.*}} : f32 into vector<1x1xf32>

  // CHECK: [[d_frag:%.+]] = nvgpu.mma.sync([[a_frag]], [[b_frag]], [[c_frag]])
  // CHECK-SAME: mmaShape = [16, 8, 4]
  // CHECK-SAME: -> vector<2x2xf32>
  %A = vector.transfer_read %arg0[%c1, %c3], %cst {in_bounds = [true, true]} : memref<20x20xf32, 3>, vector<16x4xf32>
  %B = vector.transfer_read %arg1[%c3, %c3], %cst {permutation_map = #map0, in_bounds = [true, true]} : memref<20x20xf32, 3>, vector<8x4xf32>  
  %D = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %cst_0 : vector<16x4xf32>, vector<8x4xf32> into vector<16x8xf32>

  // CHECK: vector.extract [[d_frag]][0] : vector<2x2xf32>
  // CHECK: affine.apply [[$rowC_map]]
  // CHECK: affine.apply [[$colC_map]]
  // CHECK: vector.store
  // CHECK: vector.extract [[d_frag]][1] : vector<2x2xf32>
  // CHECK: affine.apply [[$rowC8_map]]
  // CHECK: affine.apply [[$colC_map]]
  // CHECK: vector.store
  vector.transfer_write %D, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x8xf32>, memref<20x20xf32>
  return
}

// -----

#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-DAG: [[$rowA_map:#.+]] = affine_map<()[s0] -> (s0 mod 16 + 1)>
// CHECK-DAG: [[$colA_map:#.+]] = affine_map<()[s0] -> ((s0 floordiv 16) * 4 + 3)>

// CHECK-DAG: [[$rowB_map:#.+]] = affine_map<()[s0] -> (s0 mod 4 + 3)>
// CHECK-DAG: [[$colB_map:#.+]] = affine_map<()[s0] -> (s0 floordiv 4 + 3)>

// CHECK-DAG: [[$rowC_map:#.+]] = affine_map<()[s0] -> (s0 floordiv 4)>
// CHECK-DAG: [[$rowC8_map:#.+]] = affine_map<()[s0] -> (s0 floordiv 4 + 8)>
// CHECK-DAG: [[$colC_map:#.+]] = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 4) * 8)>

// CHECK-LABEL: func @m16n8k8_tf32_f32_row_row_row
func.func @m16n8k8_tf32_f32_row_row_row(%arg0: memref<20x20xf32, 3>, %arg1: memref<20x20xf32, 3>, %arg2: memref<20x20xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : vector<16x8xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %cst = arith.constant 0.000000e+00 : f32

  // CHECK: [[c_frag:%.+]] = arith.constant {{.*}} : vector<2x2xf32>

  // CHECK-DAG: [[row:%.+]] = affine.apply [[$rowA_map]]
  // CHECK-DAG: [[col:%.+]] = affine.apply [[$colA_map]]
  // CHECK: [[a_frag:%.+]] = nvgpu.ldmatrix %arg0[[[row]], [[col]]] {numTiles = 4 : i32, transpose = false}

  // b and c are not loaded by ldmatrix in this test.
  // CHECK-NOT: nvgpu.ldmatrix

  // CHECK-DAG: [[row:%.+]] = affine.apply [[$rowB_map]]
  // CHECK-DAG: [[col:%.+]] = affine.apply [[$colB_map]]
  // CHECK: [[b_el0:%.+]] = memref.load {{%.+}} : memref<20x20xf32, 3>
  // CHECK: [[b_frag0:%.+]] = vector.insert [[b_el0]], {{.*}} : f32 into vector<2x1xf32>
  // CHECK: [[b_el1:%.+]] = memref.load {{%.+}} : memref<20x20xf32, 3>
  // CHECK: [[b_frag1:%.+]] = vector.insert [[b_el1]], {{.*}} : f32 into vector<2x1xf32>

  // CHECK: [[d_frag:%.+]] = nvgpu.mma.sync([[a_frag]], [[b_frag1]], [[c_frag]])
  // CHECK-SAME: mmaShape = [16, 8, 8]
  // CHECK-SAME: -> vector<2x2xf32>
  %A = vector.transfer_read %arg0[%c1, %c3], %cst {in_bounds = [true, true]} : memref<20x20xf32, 3>, vector<16x8xf32>
  %B = vector.transfer_read %arg1[%c3, %c3], %cst {permutation_map = #map0, in_bounds = [true, true]} : memref<20x20xf32, 3>, vector<8x8xf32>
  %D = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %cst_0 : vector<16x8xf32>, vector<8x8xf32> into vector<16x8xf32>

  // CHECK: vector.extract [[d_frag]][0] : vector<2x2xf32>
  // CHECK: affine.apply [[$rowC_map]]
  // CHECK: affine.apply [[$colC_map]]
  // CHECK: vector.store
  // CHECK: vector.extract [[d_frag]][1] : vector<2x2xf32>
  // CHECK: affine.apply [[$rowC8_map]]
  // CHECK: affine.apply [[$colC_map]]
  // CHECK: vector.store
  vector.transfer_write %D, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x8xf32>, memref<20x20xf32>
  return
}

// -----

//#########################################################
// INT4 row-col-row
//#########################################################

// CHECK-DAG: [[$rowA0_map:#.+]] = affine_map<()[s0] -> (s0 mod 16)>
// CHECK-DAG: [[$colA0_map:#.+]] = affine_map<()[s0] -> ((s0 floordiv 16) * 32)>
// CHECK-DAG: [[$rowB0_map:#.+]] = affine_map<()[s0] -> (s0 mod 8)>
// CHECK-DAG: [[$colB0_map:#.+]] = affine_map<()[s0] -> ((s0 floordiv 8) * 32)>
// CHECK-DAG: [[$rowC0_map:#.+]] = affine_map<()[s0] -> (s0 floordiv 4)>
// CHECK-DAG: [[$colC0_map:#.+]] = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 4) * 8
// CHECK-DAG: [[$rowC8_map:#.+]] = affine_map<()[s0] -> (s0 floordiv 4 + 8)>

#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @m16n8k64_int4_row_col_row
func.func @m16n8k64_int4_row_col_row(%arg0: memref<128x128xi4, 3>, %arg1: memref<128x128xi4, 3>, %arg2: memref<128x128xi32>) {
  %cst  = arith.constant 0 : i4
  %cst0  = arith.constant 0 : i32
  %cst_0 = arith.constant dense<0> : vector<32x8xi4>
  %c0 = arith.constant 0 : index

  // CHECK: [[lane:%.+]] = gpu.lane_id
  // CHECK: [[row:%.+]] = affine.apply [[$rowA0_map]]()[[[lane]]]
  // CHECK: [[col:%.+]] = affine.apply [[$colA0_map]]()[[[lane]]]
  // CHECK: nvgpu.ldmatrix %arg0[[[row]], [[col]]] {numTiles = 4 : i32, transpose = false} : memref<128x128xi4, 3> -> vector<4x8xi4>

  // CHECK: [[lane:%.+]] = gpu.lane_id
  // CHECK: [[row:%.+]] = affine.apply [[$rowB0_map]]()[[[lane]]]
  // CHECK: [[col:%.+]] = affine.apply [[$colB0_map]]()[[[lane]]]
  // CHECK: nvgpu.ldmatrix %arg1[[[row]], [[col]]] {numTiles = 2 : i32, transpose = false} : memref<128x128xi4, 3> -> vector<2x8xi4>
  
  // CHECK: [[lane:%.+]] = gpu.lane_id
  // CHECK: [[row:%.+]] = affine.apply [[$rowC0_map]]()[{{%.+}}]
  // CHECK: [[col:%.+]] = affine.apply [[$colC0_map]]()[{{%.+}}]
  // CHECK: vector.load %arg2[[[row]], [[col]]] : memref<128x128xi32>, vector<2xi32>

  // CHECK: [[row:%.+]] = affine.apply [[$rowC8_map]]()[{{%.+}}]
  // CHECK: [[col:%.+]] = affine.apply [[$colC0_map]]()[{{%.+}}]
  // CHECK: vector.load %arg2[[[row]], [[col]]] : memref<128x128xi32>, vector<2xi32>
  // CHECK-NOT: vector.load

  %A = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<128x128xi4, 3>, vector<16x64xi4>
  %B = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : memref<128x128xi4, 3>, vector<8x64xi4>
  %C = vector.transfer_read %arg2[%c0, %c0], %cst0 {in_bounds = [true, true]} : memref<128x128xi32>, vector<16x8xi32>
  // CHECK: [[d:%.+]] = nvgpu.mma.sync({{.*}}) {mmaShape = [16, 8, 64]} : (vector<4x8xi4>, vector<2x8xi4>, vector<2x2xi32>) -> vector<2x2xi32>
  %D = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<16x64xi4>, vector<8x64xi4> into vector<16x8xi32>

  // CHECK: [[lane:%.+]] = gpu.lane_id
  // CHECK: [[v:%.+]] = vector.extract [[d]][0] : vector<2x2xi32>  
  // CHECK: [[row:%.+]] = affine.apply [[$rowC0_map]]()[[[lane]]]
  // CHECK: [[col:%.+]] = affine.apply [[$colC0_map]]()[[[lane]]]
  // CHECK: vector.store [[v]], %arg2[[[row]], [[col]]] : memref<128x128xi32>, vector<2xi32>
  
  // CHECK: [[v:%.+]] = vector.extract [[d]][1] : vector<2x2xi32>
  // CHECK: [[row:%.+]] = affine.apply [[$rowC8_map]]()[[[lane]]]
  // CHECK: [[col:%.+]] = affine.apply [[$colC0_map]]()[[[lane]]]
  // CHECK: vector.store [[v]], %arg2[[[row]], [[col]]] : memref<128x128xi32>, vector<2xi32>
  vector.transfer_write %D, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x8xi32>, memref<128x128xi32>
  return
}

// -----

//#########################################################
// INT8 row-col-row
//#########################################################

// CHECK-DAG: [[$rowA0_map:#.+]] = affine_map<()[s0] -> (s0 mod 16)>
// CHECK-DAG: [[$colA0_map:#.+]] = affine_map<()[s0] -> ((s0 floordiv 16) * 16)>
// CHECK-DAG: [[$rowB0_map:#.+]] = affine_map<()[s0] -> (s0 mod 8)>
// CHECK-DAG: [[$colB0_map:#.+]] = affine_map<()[s0] -> ((s0 floordiv 8) * 16)>

// CHECK-DAG: [[$rowC0_map:#.+]] = affine_map<()[s0] -> (s0 floordiv 4)>
// CHECK-DAG: [[$colC0_map:#.+]] = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 4) * 8)>
// CHECK-DAG: [[$rowC8_map:#.+]] = affine_map<()[s0] -> (s0 floordiv 4 + 8)>


#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @m16n8k32_int8_row_col_row
func.func @m16n8k32_int8_row_col_row(%arg0: memref<128x128xi8, 3>, %arg1: memref<128x128xi8, 3>, %arg2: memref<128x128xi32>) {
  %cst_0 = arith.constant dense<0> : vector<32x8xi8>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0 : i8
  %cst0 = arith.constant 0 : i32

  // CHECK: [[lane:%.+]] = gpu.lane_id
  // CHECK: [[row:%.+]] = affine.apply [[$rowA0_map]]()[[[lane]]]
  // CHECK: [[col:%.+]] = affine.apply [[$colA0_map]]()[[[lane]]]
  // CHECK: nvgpu.ldmatrix %arg0[[[row]], [[col]]] {numTiles = 4 : i32, transpose = false} : memref<128x128xi8, 3> -> vector<4x4xi8>

  // CHECK: [[lane:%.+]] = gpu.lane_id
  // CHECK: [[row:%.+]] = affine.apply [[$rowB0_map]]()[[[lane]]]
  // CHECK: [[col:%.+]] = affine.apply [[$colB0_map]]()[[[lane]]]
  // CHECK: nvgpu.ldmatrix %arg1[[[row]], [[col]]] {numTiles = 2 : i32, transpose = false} : memref<128x128xi8, 3> -> vector<2x4xi8>
  
  // CHECK: [[lane:%.+]] = gpu.lane_id
  // CHECK: [[row:%.+]] = affine.apply [[$rowC0_map]]()[[[lane]]]
  // CHECK: [[col:%.+]] = affine.apply [[$colC0_map]]()[[[lane]]]
  // CHECK: vector.load %arg2[[[row]], [[col]]] : memref<128x128xi32>, vector<2xi32>
  // CHECK: [[row:%.+]] = affine.apply [[$rowC8_map]]()[[[lane]]]
  // CHECK: [[col:%.+]] = affine.apply [[$colC0_map]]()[[[lane]]]
  // CHECK: vector.load %arg2[[[row]], [[col]]] : memref<128x128xi32>, vector<2xi32>
  // CHECK-NOT: vector.load %arg2{{.*}}

  %A = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<128x128xi8, 3>, vector<16x32xi8>
  %B = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : memref<128x128xi8, 3>, vector<8x32xi8>
  %C = vector.transfer_read %arg2[%c0, %c0], %cst0 {in_bounds = [true, true]} : memref<128x128xi32>, vector<16x8xi32>
  // CHECK: [[d:%.+]] = nvgpu.mma.sync({{.*}}) {mmaShape = [16, 8, 32]} : (vector<4x4xi8>, vector<2x4xi8>, vector<2x2xi32>) -> vector<2x2xi32>
  %D = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<16x32xi8>, vector<8x32xi8> into vector<16x8xi32>

  // CHECK: [[lane:%.+]] = gpu.lane_id
  // CHECK: [[v:%.+]] = vector.extract [[d]][0] : vector<2x2xi32>  
  // CHECK: [[row:%.+]] = affine.apply [[$rowC0_map]]()[[[lane]]]
  // CHECK: [[col:%.+]] = affine.apply [[$colC0_map]]()[[[lane]]]
  // CHECK: vector.store [[v]], %arg2[[[row]], [[col]]] : memref<128x128xi32>, vector<2xi32>
  // CHECK: [[v:%.+]] = vector.extract [[d]][1] : vector<2x2xi32>
  // CHECK: [[row:%.+]] = affine.apply [[$rowC8_map]]()[[[lane]]]
  // CHECK: [[col:%.+]] = affine.apply [[$colC0_map]]()[[[lane]]]
  // CHECK: vector.store [[v]], %arg2[[[row]], [[col]]] : memref<128x128xi32>, vector<2xi32>
  vector.transfer_write %D, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x8xi32>, memref<128x128xi32>
  return
}

// -----

#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-DAG: [[$rowA0_map:#.+]] = affine_map<()[s0] -> (s0 floordiv 4)>
// CHECK-DAG: [[$colA0_map:#.+]] = affine_map<()[s0] -> (s0 mod 4)>
// CHECK-DAG: [[$rowA8_map:#.+]] = affine_map<()[s0] -> (s0 floordiv 4 + 8)>
// CHECK-DAG: [[$colA4_map:#.+]] = affine_map<()[s0] -> (s0 mod 4 + 4)>

// CHECK-DAG: [[$rowB0_map:#.+]] = affine_map<()[s0] -> (s0 mod 8)>
// CHECK-DAG: [[$colB0_map:#.+]] = affine_map<()[s0] -> ((s0 floordiv 8) * 4)>

// CHECK-DAG: [[$rowC_map:#.+]] = affine_map<()[s0] -> (s0 floordiv 4 + 16)>
// CHECK-DAG: [[$rowC8_map:#.+]] = affine_map<()[s0] -> (s0 floordiv 4 + 24)>
// CHECK-DAG: [[$colC_map:#.+]] = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 4) * 8 + 8)>

// CHECK-LABEL: func @m16n8k8_tf32_f32_col_col_row
func.func @m16n8k8_tf32_f32_col_col_row(%arg0: memref<20x20xf32, 3>, %arg1: memref<20x20xf32, 3>, %arg2: memref<20x20xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : vector<16x8xf32>
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %cst = arith.constant 0.000000e+00 : f32

  // CHECK: [[c_frag:%.+]] = arith.constant {{.*}} : vector<2x2xf32>

  // CHECK-DAG: [[row:%.+]] = affine.apply [[$rowA0_map]]
  // CHECK-DAG: [[col:%.+]] = affine.apply [[$colA0_map]]
  // CHECK: [[a_el0:%.+]] = memref.load {{%.+}} : memref<20x20xf32, 3>
  // CHECK: [[a_frag0:%.+]] = vector.insert [[a_el0]], {{.*}} [0, 0] : f32 into vector<4x1xf32>

  // CHECK-DAG: [[row:%.+]] = affine.apply [[$rowA8_map]]
  // CHECK-DAG: [[col:%.+]] = affine.apply [[$colA0_map]]
  // CHECK: [[a_el0:%.+]] = memref.load {{%.+}} : memref<20x20xf32, 3>
  // CHECK: [[a_frag0:%.+]] = vector.insert [[a_el0]], {{.*}} [1, 0] : f32 into vector<4x1xf32>

  // CHECK: [[a_el:%.+]] = memref.load {{%.+}} : memref<20x20xf32, 3>
  // CHECK: [[a_frag:%.+]] = vector.insert [[a_el]], {{.*}} [2, 0] : f32 into vector<4x1xf32>
  // CHECK: [[a_el:%.+]] = memref.load {{%.+}} : memref<20x20xf32, 3>
  // CHECK: [[a_frag:%.+]] = vector.insert [[a_el]], {{.*}} [3, 0] : f32 into vector<4x1xf32>

  // CHECK-DAG: [[row:%.+]] = affine.apply [[$rowB0_map]]
  // CHECK-DAG: [[col:%.+]] = affine.apply [[$colB0_map]]
  // CHECK: [[b_frag:%.+]] = nvgpu.ldmatrix %arg1[[[row]], [[col]]] {numTiles = 2 : i32, transpose = false}

  // CHECK: [[d_frag:%.+]] = nvgpu.mma.sync([[a_frag]], [[b_frag]], [[c_frag]])
  // CHECK-SAME: mmaShape = [16, 8, 8]
  // CHECK-SAME: -> vector<2x2xf32>
  %A = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true], permutation_map = #map0} : memref<20x20xf32, 3>, vector<16x8xf32>
  %B = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : memref<20x20xf32, 3>, vector<8x8xf32>
  %D = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], 
    kind = #vector.kind<add>} %A, %B, %cst_0 : vector<16x8xf32>, vector<8x8xf32> into vector<16x8xf32>

  // CHECK: vector.extract [[d_frag]][0] : vector<2x2xf32>
  // CHECK: affine.apply [[$rowC_map]]
  // CHECK: affine.apply [[$colC_map]]
  // CHECK: vector.store
  // CHECK: vector.extract [[d_frag]][1] : vector<2x2xf32>
  // CHECK: affine.apply [[$rowC8_map]]
  // CHECK: affine.apply [[$colC_map]]
  // CHECK: vector.store
  vector.transfer_write %D, %arg2[%c16, %c8] {in_bounds = [true, true]} : vector<16x8xf32>, memref<20x20xf32>
  return
}