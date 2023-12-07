// RUN: mlir-opt %s -test-transform-dialect-interpreter -split-input-file | FileCheck %s

// CHECK: #[[$div4:.*]]  = affine_map<()[s0] -> (s0 floordiv 4)>                                    
// CHECK: #[[$mod4:.*]] = affine_map<()[s0] -> (s0 mod 4)>
// CHECK: #[[$div4p8:.*]] = affine_map<()[s0] -> (s0 floordiv 4 + 8)>
// CHECK: #[[$map3:.*]] = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 4) * 8)>
// CHECK: #[[$map4:.*]] = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 4) * 8 + 1)>

// CHECK-LABEL: func.func @matmul_16x8x4xf32_global
func.func @matmul_16x8x4xf32_global(
    %A: memref<16x4xf32>, %B: memref<4x8xf32>, %C: memref<16x8xf32>) {
// CHECK-SAME:                                        %[[VAL_0:.*]]: memref<16x4xf32>,
// CHECK-SAME:                                        %[[VAL_1:.*]]: memref<4x8xf32>,
// CHECK-SAME:                                        %[[VAL_2:.*]]: memref<16x8xf32>) {

// CHECK:           %[[TIDX:.*]] = gpu.thread_id  x
// CHECK:           %[[VAL_4:.*]] = affine.apply #[[$div4]]()[%[[TIDX]]]
// CHECK:           %[[VAL_5:.*]] = affine.apply #[[$mod4]]()[%[[TIDX]]]
// CHECK:           %[[VAL_6:.*]] = memref.load %[[VAL_0]][%[[VAL_4]], %[[VAL_5]]] : memref<16x4xf32>
// CHECK:           %[[VAL_7:.*]] = affine.apply #[[$div4p8]]()[%[[TIDX]]]
// CHECK:           %[[VAL_8:.*]] = affine.apply #[[$mod4]]()[%[[TIDX]]]
// CHECK:           %[[VAL_9:.*]] = memref.load %[[VAL_0]][%[[VAL_7]], %[[VAL_8]]] : memref<16x4xf32>
// CHECK:           %[[VAL_10:.*]] = vector.splat %[[VAL_6]] : vector<2x1xf32>
// CHECK:           %[[VAL_11:.*]] = vector.insert %[[VAL_6]], %[[VAL_10]] [0, 0] : f32 into vector<2x1xf32>
// CHECK:           %[[LHS:.*]] = vector.insert %[[VAL_9]], %[[VAL_11]] [1, 0] : f32 into vector<2x1xf32>
//
// CHECK:           %[[VAL_13:.*]] = affine.apply #[[$mod4]]()[%[[TIDX]]]
// CHECK:           %[[VAL_14:.*]] = affine.apply #[[$div4]]()[%[[TIDX]]]
// CHECK:           %[[VAL_15:.*]] = memref.load %[[VAL_1]][%[[VAL_13]], %[[VAL_14]]] : memref<4x8xf32>
// CHECK:           %[[VAL_16:.*]] = vector.splat %[[VAL_15]] : vector<1x1xf32>
// CHECK:           %[[RHS:.*]] = vector.insert %[[VAL_15]], %[[VAL_16]] [0, 0] : f32 into vector<1x1xf32>
//
// CHECK:           %[[VAL_18:.*]] = affine.apply #[[$div4]]()[%[[TIDX]]]
// CHECK:           %[[VAL_19:.*]] = affine.apply #[[$map3]]()[%[[TIDX]]]
// CHECK:           %[[VAL_20:.*]] = memref.load %[[VAL_2]][%[[VAL_18]], %[[VAL_19]]] : memref<16x8xf32>
// CHECK:           %[[VAL_21:.*]] = affine.apply #[[$div4]]()[%[[TIDX]]]
// CHECK:           %[[VAL_22:.*]] = affine.apply #[[$map4]]()[%[[TIDX]]]
// CHECK:           %[[VAL_23:.*]] = memref.load %[[VAL_2]][%[[VAL_21]], %[[VAL_22]]] : memref<16x8xf32>
// CHECK:           %[[VAL_24:.*]] = affine.apply #[[$div4p8]]()[%[[TIDX]]]
// CHECK:           %[[VAL_25:.*]] = affine.apply #[[$map3]]()[%[[TIDX]]]
// CHECK:           %[[VAL_26:.*]] = memref.load %[[VAL_2]][%[[VAL_24]], %[[VAL_25]]] : memref<16x8xf32>
// CHECK:           %[[VAL_27:.*]] = affine.apply #[[$div4p8]]()[%[[TIDX]]]
// CHECK:           %[[VAL_28:.*]] = affine.apply #[[$map4]]()[%[[TIDX]]]
// CHECK:           %[[VAL_29:.*]] = memref.load %[[VAL_2]][%[[VAL_27]], %[[VAL_28]]] : memref<16x8xf32>
// CHECK:           %[[VAL_30:.*]] = vector.splat %[[VAL_20]] : vector<2x2xf32>
// CHECK:           %[[VAL_31:.*]] = vector.insert %[[VAL_20]], %[[VAL_30]] [0, 0] : f32 into vector<2x2xf32>
// CHECK:           %[[VAL_32:.*]] = vector.insert %[[VAL_23]], %[[VAL_31]] [0, 1] : f32 into vector<2x2xf32>
// CHECK:           %[[VAL_33:.*]] = vector.insert %[[VAL_26]], %[[VAL_32]] [1, 0] : f32 into vector<2x2xf32>
// CHECK:           %[[RES:.*]] = vector.insert %[[VAL_29]], %[[VAL_33]] [1, 1] : f32 into vector<2x2xf32>
//
// CHECK:           %[[VAL_35:.*]] = nvgpu.mma.sync(%[[LHS]], %[[RHS]], %[[RES]]) {mmaShape = [16, 8, 4], tf32Enabled} : (vector<2x1xf32>, vector<1x1xf32>, vector<2x2xf32>) -> vector<2x2xf32>
//
// CHECK:           %[[VAL_36:.*]] = vector.extract %[[VAL_35]][0, 0] : vector<2x2xf32>
// CHECK:           %[[VAL_37:.*]] = vector.extract %[[VAL_35]][0, 1] : vector<2x2xf32>
// CHECK:           %[[VAL_38:.*]] = vector.extract %[[VAL_35]][1, 0] : vector<2x2xf32>
// CHECK:           %[[VAL_39:.*]] = vector.extract %[[VAL_35]][1, 1] : vector<2x2xf32>
// CHECK:           %[[VAL_40:.*]] = affine.apply #[[$div4]]()[%[[TIDX]]]
// CHECK:           %[[VAL_41:.*]] = affine.apply #[[$map3]]()[%[[TIDX]]]
// CHECK:           memref.store %[[VAL_36]], %[[VAL_2]][%[[VAL_40]], %[[VAL_41]]] : memref<16x8xf32>
// CHECK:           %[[VAL_42:.*]] = affine.apply #[[$div4]]()[%[[TIDX]]]
// CHECK:           %[[VAL_43:.*]] = affine.apply #[[$map4]]()[%[[TIDX]]]
// CHECK:           memref.store %[[VAL_37]], %[[VAL_2]][%[[VAL_42]], %[[VAL_43]]] : memref<16x8xf32>
// CHECK:           %[[VAL_44:.*]] = affine.apply #[[$div4p8]]()[%[[TIDX]]]
// CHECK:           %[[VAL_45:.*]] = affine.apply #[[$map3]]()[%[[TIDX]]]
// CHECK:           memref.store %[[VAL_38]], %[[VAL_2]][%[[VAL_44]], %[[VAL_45]]] : memref<16x8xf32>
// CHECK:           %[[VAL_46:.*]] = affine.apply #[[$div4p8]]()[%[[TIDX]]]
// CHECK:           %[[VAL_47:.*]] = affine.apply #[[$map4]]()[%[[TIDX]]]
// CHECK:           memref.store %[[VAL_39]], %[[VAL_2]][%[[VAL_46]], %[[VAL_47]]] : memref<16x8xf32>
// CHECK:           return
// CHECK:         }
  linalg.matmul ins(%A, %B: memref<16x4xf32>, memref<4x8xf32>)
            outs(%C: memref<16x8xf32>)
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1 
    : (!transform.any_op) -> !transform.any_op
  transform.nvgpu.rewrite_matmul_as_mma_sync %matmul 
    : (!transform.any_op) -> ()
}

// -----

// CHECK-LABEL: func.func @matmul_16x8x16xf16_global
func.func @matmul_16x8x16xf16_global(
    %A: memref<16x16xf16>, %B: memref<16x8xf16>, %C: memref<16x8xf16>) {

  // CHECK-COUNT-8: memref.load {{.*}} : memref<16x16xf16>
  // CHECK-COUNT-8: vector.insert {{.*}} : f16 into vector<4x2xf16> 
  // CHECK-COUNT-4: memref.load {{.*}} : memref<16x8xf16>
  // CHECK-COUNT-4: vector.insert {{.*}} : f16 into vector<2x2xf16> 
  // CHECK-COUNT-4: memref.load {{.*}} : memref<16x8xf16>
  // CHECK-COUNT-4: vector.insert {{.*}} : f16 into vector<2x2xf16>
  //
  //         CHECK: nvgpu.mma.sync(%{{.*}}) {mmaShape = [16, 8, 16]} 
  //    CHECK-SAME:   : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
  //
  // CHECK-COUNT-4: vector.extract %{{.*}} : vector<2x2xf16>
  // CHECK-COUNT-4: memref.store %{{.*}} : memref<16x8xf16>
  linalg.matmul ins(%A, %B: memref<16x16xf16>, memref<16x8xf16>)
            outs(%C: memref<16x8xf16>)
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1 
    : (!transform.any_op) -> !transform.any_op
  transform.nvgpu.rewrite_matmul_as_mma_sync %matmul 
    : (!transform.any_op) -> ()
}
