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
