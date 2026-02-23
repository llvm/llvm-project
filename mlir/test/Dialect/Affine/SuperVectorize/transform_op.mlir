//RUN: mlir-opt %s --transform-interpreter -split-input-file | FileCheck %s

// CHECK-LABEL: func @vec1d_3
func.func @vec1d_3(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
   %c0 = arith.constant 0 : index
   %c1 = arith.constant 1 : index
   %c2 = arith.constant 2 : index
   %M = memref.dim %A, %c0 : memref<?x?xf32>
   %N = memref.dim %A, %c1 : memref<?x?xf32>
   %P = memref.dim %B, %c2 : memref<?x?x?xf32>

// CHECK:   %{{.*}} = vector.transfer_read {{.*}} : memref<?x?xf32>, vector<128xf32>
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

// CHECK-LABEL: func @vectorize_matmul
func.func @vectorize_matmul(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = memref.dim %arg0, %c0 : memref<?x?xf32>
  %K = memref.dim %arg0, %c1 : memref<?x?xf32>
  %N = memref.dim %arg2, %c1 : memref<?x?xf32>
  // CHECK:     vector.transfer_write {{.*}} : vector<4x8xf32>, memref<?x?xf32>
  affine.for %i0 = affine_map<(d0) -> (d0)>(%c0) to affine_map<(d0) -> (d0)>(%M) {
    affine.for %i1 = affine_map<(d0) -> (d0)>(%c0) to affine_map<(d0) -> (d0)>(%N) {
      %cst = arith.constant 0.000000e+00 : f32
      affine.store %cst, %arg2[%i0, %i1] : memref<?x?xf32>
    }
  }
  //      CHECK:        %{{.*}} = vector.transfer_read {{.*}} : memref<?x?xf32>, vector<4x8xf32>
  //      CHECK:        %{{.*}} = vector.transfer_read {{.*}} : memref<?x?xf32>, vector<4x8xf32>
  //      CHECK:        %{{.*}} = vector.transfer_read {{.*}} : memref<?x?xf32>, vector<4x8xf32>
  //      CHECK:        vector.transfer_write %{{.*}} : vector<4x8xf32>, memref<?x?xf32>
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
  // CHECK:           %{{.*}} = vector.transfer_read {{.*}} : memref<?x?x?xf32>, vector<32x64x256xf32>
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

// CHECK-LABEL: @vecdim_reduction_minf
// CHECK:           %{{.*}} = vector.transfer_read {{.*}} : memref<256x512xf32>, vector<128xf32>
// CHECK:         %{{.*}} = vector.reduction <minimumf>, {{.*}} : vector<128xf32> into f32

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["affine.for"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.affine.super_vectorize %0 [128] vectorize_reductions=true : !transform.any_op
    transform.yield
  }
}
