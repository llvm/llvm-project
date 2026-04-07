// RUN: mlir-opt %s -affine-super-vectorizer-test=vectorize-affine-loop-nest -split-input-file 2>&1 |  FileCheck %s

func.func @unparallel_loop_reduction_unsupported(%in: memref<256x512xf32>, %out: memref<256xf32>) {
 // CHECK: Outermost loop cannot be parallel
 %cst = arith.constant 1.000000e+00 : f32
 %final_red = affine.for %j = 0 to 512 iter_args(%red_iter = %cst) -> (f32) {
   %add = arith.addf %red_iter, %red_iter : f32
   affine.yield %add : f32
 }
 return
}

// -----

#map = affine_map<(d0)[s0] -> (d0 mod s0)>
#map1 = affine_map<(d0)[s0] -> (d0 floordiv s0)>

func.func @iv_mapped_to_multiple_indices_unsupported(%arg0: index) -> memref<2x2xf32> {
  %c2 = arith.constant 2 : index
  %cst = arith.constant 1.0 : f32
  %alloc = memref.alloc() : memref<2x2xf32>
    
    affine.for %i = 0 to 4 {
      %row = affine.apply #map1(%i)[%c2]  
      %col = affine.apply #map(%i)[%c2]  
      affine.store %cst, %alloc[%row, %col] : memref<2x2xf32>
    }
    
    return %alloc : memref<2x2xf32>
  }

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0)[s0] -> (d0 floordiv s0)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0)[s0] -> (d0 mod s0)>

// CHECK-LABEL:   func.func @iv_mapped_to_multiple_indices_unsupported(
// CHECK-SAME:      %[[VAL_0:.*]]: index) -> memref<2x2xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : index
// CHECK:           affine.for %[[VAL_4:.*]] = 0 to 4 {
// CHECK:             %[[VAL_5:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_4]]){{\[}}%[[VAL_1]]]
// CHECK:             %[[VAL_6:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_4]]){{\[}}%[[VAL_1]]]
// CHECK:           }
// CHECK:         }

// -----

// Regression test: when the store's permutation map has a broadcast dimension
// (because the index is invariant w.r.t. the vectorized loop), vectorization
// must bail out gracefully instead of emitting an invalid transfer_write.

#map_id = affine_map<(d0) -> (d0)>
#map_id_p1 = affine_map<(d0) -> (d0 + 1)>

// CHECK-LABEL: func.func @store_broadcast_perm_map_unsupported
// CHECK:         affine.for
// CHECK:           affine.for
// CHECK:             affine.load
// CHECK:             affine.store
// CHECK-NOT:     vector.transfer_write
func.func @store_broadcast_perm_map_unsupported(%arg0: memref<4x4xf32>, %arg1: memref<4xf32>) {
  affine.for %i = 0 to 4 {
    affine.for %j = #map_id(%i) to #map_id_p1(%i) {
      %0 = affine.load %arg0[%j, %j] : memref<4x4xf32>
      affine.store %0, %arg1[%j] : memref<4xf32>
    }
  }
  return
}

// -----

// Regression test for https://github.com/llvm/llvm-project/issues/128334
// The vectorizer test utility used to crash when a reduction loop with a
// dynamic upper bound was vectorized via 'vectorizeAffineLoopNest', because
// the reduction descriptors were not added to the vectorization strategy.

// CHECK-LABEL: func.func @reduction_loop_dynamic_bound_vectorized
// CHECK:         affine.for %{{.*}} iter_args(%{{.*}} = {{.*}}) -> (vector<4xf32>) {
// CHECK:           vector.transfer_read
// CHECK:           arith.addf
// CHECK:           affine.yield
// CHECK:         }
// CHECK:         vector.reduction <add>
func.func @reduction_loop_dynamic_bound_vectorized(%buffer: memref<1024xf32>) {
  %c10 = arith.constant 10 : index
  %sum_0 = arith.constant 0.0 : f32
  affine.for %i = 0 to %c10 iter_args(%sum_iter = %sum_0) -> (f32) {
    %t = affine.load %buffer[%i] : memref<1024xf32>
    %sum_next = arith.addf %sum_iter, %t : f32
    affine.yield %sum_next : f32
  }
  return
}
