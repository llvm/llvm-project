//RUN: mlir-opt %s --transform-interpreter -split-input-file | FileCheck %s

// CHECK-LABEL: func @vec1d_3
func.func @vec1d_3(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: [[ARG_M:%[0-9a-zA-Z_]+]] = memref.dim %arg0, %[[C0]] : memref<?x?xf32>
// CHECK-DAG: [[ARG_N:%[0-9a-zA-Z_]+]] = memref.dim %arg0, %[[C1]] : memref<?x?xf32>
// CHECK-DAG: [[ARG_P:%[0-9a-zA-Z_]+]] = memref.dim %arg1, %[[C2]] : memref<?x?x?xf32>
   %c0 = arith.constant 0 : index
   %c1 = arith.constant 1 : index
   %c2 = arith.constant 2 : index
   %M = memref.dim %A, %c0 : memref<?x?xf32>
   %N = memref.dim %A, %c1 : memref<?x?xf32>
   %P = memref.dim %B, %c2 : memref<?x?x?xf32>

// CHECK:for [[IV8:%[0-9a-zA-Z_]+]] = 0 to [[ARG_M]] step 128
// CHECK-NEXT:   for [[IV9:%[0-9a-zA-Z_]*]] = 0 to [[ARG_N]] {
// CHECK-NEXT:   %[[APP9_0:[0-9a-zA-Z_]+]] = affine.apply {{.*}}([[IV9]], [[IV8]])
// CHECK-NEXT:   %[[APP9_1:[0-9a-zA-Z_]+]] = affine.apply {{.*}}([[IV9]], [[IV8]])
// CHECK-NEXT:   %[[CST:.*]] = arith.constant 0.0{{.*}}: f32
// CHECK-NEXT:   {{.*}} = vector.transfer_read %{{.*}}[%[[APP9_0]], %[[APP9_1]]], %[[CST]] : memref<?x?xf32>, vector<128xf32>
   affine.for %i8 = 0 to %M { // vectorized
     affine.for %i9 = 0 to %N {
       %a9 = affine.load %A[%i9, %i8 + %i9] : memref<?x?xf32>
     }
   }
   return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["affine.for"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.affine.super_vectorize %0 [128] : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-DAG: #[[$map_id1:map[0-9]*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[$map_proj_d0d1_zerod1:map[0-9]*]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK-DAG: #[[$map_proj_d0d1_d0zero:map[0-9]*]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK-LABEL: func @vectorize_matmul
func.func @vectorize_matmul(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = memref.dim %arg0, %c0 : memref<?x?xf32>
  %K = memref.dim %arg0, %c1 : memref<?x?xf32>
  %N = memref.dim %arg2, %c1 : memref<?x?xf32>
  //      CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-NEXT: %[[M:.*]] = memref.dim %{{.*}}, %[[C0]] : memref<?x?xf32>
  // CHECK-NEXT: %[[K:.*]] = memref.dim %{{.*}}, %[[C1]] : memref<?x?xf32>
  // CHECK-NEXT: %[[N:.*]] = memref.dim %{{.*}}, %[[C1]] : memref<?x?xf32>
  //      CHECK: {{.*}} #[[$map_id1]](%[[M]]) step 4 {
  // CHECK-NEXT:   {{.*}} #[[$map_id1]](%[[N]]) step 8 {
  //      CHECK:     %[[VC0:.*]] = arith.constant dense<0.000000e+00> : vector<4x8xf32>
  // CHECK-NEXT:     vector.transfer_write %[[VC0]], %{{.*}}[%{{.*}}, %{{.*}}] : vector<4x8xf32>, memref<?x?xf32>
  affine.for %i0 = affine_map<(d0) -> (d0)>(%c0) to affine_map<(d0) -> (d0)>(%M) {
    affine.for %i1 = affine_map<(d0) -> (d0)>(%c0) to affine_map<(d0) -> (d0)>(%N) {
      %cst = arith.constant 0.000000e+00 : f32
      affine.store %cst, %arg2[%i0, %i1] : memref<?x?xf32>
    }
  }
  //      CHECK:  affine.for %[[I2:.*]] = #[[$map_id1]](%[[C0]]) to #[[$map_id1]](%[[M]]) step 4 {
  // CHECK-NEXT:    affine.for %[[I3:.*]] = #[[$map_id1]](%[[C0]]) to #[[$map_id1]](%[[N]]) step 8 {
  // CHECK-NEXT:      affine.for %[[I4:.*]] = #[[$map_id1]](%[[C0]]) to #[[$map_id1]](%[[K]]) {
  //      CHECK:        %[[A:.*]] = vector.transfer_read %{{.*}}[%[[I4]], %[[I3]]], %{{.*}} {permutation_map = #[[$map_proj_d0d1_zerod1]]} : memref<?x?xf32>, vector<4x8xf32>
  //      CHECK:        %[[B:.*]] = vector.transfer_read %{{.*}}[%[[I2]], %[[I4]]], %{{.*}} {permutation_map = #[[$map_proj_d0d1_d0zero]]} : memref<?x?xf32>, vector<4x8xf32>
  // CHECK-NEXT:        %[[C:.*]] = arith.mulf %[[B]], %[[A]] : vector<4x8xf32>
  //      CHECK:        %[[D:.*]] = vector.transfer_read %{{.*}}[%[[I2]], %[[I3]]], %{{.*}} : memref<?x?xf32>, vector<4x8xf32>
  // CHECK-NEXT:        %[[E:.*]] = arith.addf %[[D]], %[[C]] : vector<4x8xf32>
  //      CHECK:        vector.transfer_write %[[E]], %{{.*}}[%[[I2]], %[[I3]]] : vector<4x8xf32>, memref<?x?xf32>
  affine.for %i2 = affine_map<(d0) -> (d0)>(%c0) to affine_map<(d0) -> (d0)>(%M) {
    affine.for %i3 = affine_map<(d0) -> (d0)>(%c0) to affine_map<(d0) -> (d0)>(%N) {
      affine.for %i4 = affine_map<(d0) -> (d0)>(%c0) to affine_map<(d0) -> (d0)>(%K) {
        %6 = affine.load %arg1[%i4, %i3] : memref<?x?xf32>
        %7 = affine.load %arg0[%i2, %i4] : memref<?x?xf32>
        %8 = arith.mulf %7, %6 : f32
        %9 = affine.load %arg2[%i2, %i3] : memref<?x?xf32>
        %10 = arith.addf %9, %8 : f32
        affine.store %10, %arg2[%i2, %i3] : memref<?x?xf32>
      }
    }
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["affine.for"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.affine.super_vectorize %0 [4, 8] : !transform.any_op
    transform.yield
  }
}

// -----

func.func @vec3d(%A : memref<?x?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = memref.dim %A, %c0 : memref<?x?x?xf32>
  %1 = memref.dim %A, %c1 : memref<?x?x?xf32>
  %2 = memref.dim %A, %c2 : memref<?x?x?xf32>
  // CHECK: affine.for %{{.*}} = 0 to %{{.*}} {
  // CHECK:   affine.for %{{.*}} = 0 to %{{.*}} {
  // CHECK:     affine.for %{{.*}} = 0 to %{{.*}} step 32 {
  // CHECK:       affine.for %{{.*}} = 0 to %{{.*}} step 64 {
  // CHECK:         affine.for %{{.*}} = 0 to %{{.*}} step 256 {
  // CHECK:           %{{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}], %{{.*}} : memref<?x?x?xf32>, vector<32x64x256xf32>
  affine.for %t0 = 0 to %0 {
    affine.for %t1 = 0 to %0 {
      affine.for %i0 = 0 to %0 {
        affine.for %i1 = 0 to %1 {
          affine.for %i2 = 0 to %2 {
            %a2 = affine.load %A[%i0, %i1, %i2] : memref<?x?x?xf32>
          }
        }
      }
    }
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["affine.for"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.affine.super_vectorize %0 [32, 64, 256] fastest_varying_pattern=[2,1,0] : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @vecdim_reduction_minf
// CHECK:       affine.for %{{.*}} = 0 to 256 {
// CHECK:         %[[vmax:.*]] = arith.constant dense<0x7F800000> : vector<128xf32>
// CHECK:         %[[vred:.*]] = affine.for %{{.*}} = 0 to 512 step 128 iter_args(%[[red_iter:.*]] = %[[vmax]]) -> (vector<128xf32>) {
// CHECK:           %[[ld:.*]] = vector.transfer_read %{{.*}} : memref<256x512xf32>, vector<128xf32>
// CHECK:           %[[min:.*]] = arith.minimumf %[[red_iter]], %[[ld]] : vector<128xf32>
// CHECK:           affine.yield %[[min]] : vector<128xf32>
// CHECK:         }
// CHECK:         %[[final_min:.*]] = vector.reduction <minimumf>, %[[vred:.*]] : vector<128xf32> into f32
// CHECK:         affine.store %[[final_min]], %{{.*}} : memref<256xf32>
// CHECK:       }

func.func @vecdim_reduction_minf(%in: memref<256x512xf32>, %out: memref<256xf32>) {
 %cst = arith.constant 0x7F800000 : f32
 affine.for %i = 0 to 256 {
   %final_red = affine.for %j = 0 to 512 iter_args(%red_iter = %cst) -> (f32) {
     %ld = affine.load %in[%i, %j] : memref<256x512xf32>
     %min = arith.minimumf %red_iter, %ld : f32
     affine.yield %min : f32
   }
   affine.store %final_red, %out[%i] : memref<256xf32>
 }
 return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["affine.for"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.affine.super_vectorize %0 [128] vectorize_reductions=true : !transform.any_op
    transform.yield
  }
}