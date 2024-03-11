// RUN: mlir-opt %s -split-input-file -test-affine-access-analysis -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @loop_1d
func.func @loop_1d(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
   %c0 = arith.constant 0 : index
   %M = memref.dim %A, %c0 : memref<?x?xf32>
   affine.for %i = 0 to %M {
     affine.for %j = 0 to %M {
       affine.load %A[%c0, %i] : memref<?x?xf32>
       // expected-remark@above {{contiguous along loop 0}}
       affine.load %A[%c0, 8 * %i + %j] : memref<?x?xf32>
       // expected-remark@above {{contiguous along loop 1}}
       // Note/FIXME: access stride isn't being checked.
       // expected-remark@-3 {{contiguous along loop 0}}

       // These are all non-contiguous along both loops. Nothing is emitted.
       affine.load %A[%i, %c0] : memref<?x?xf32>
       // Note/FIXME: access stride isn't being checked.
       affine.load %A[%i, 8 * %j] : memref<?x?xf32>
       // expected-remark@above {{contiguous along loop 1}}
       affine.load %A[%j, 4 * %i] : memref<?x?xf32>
       // expected-remark@above {{contiguous along loop 0}}
     }
   }
   return
}

// -----

#map = affine_map<(d0) -> (d0 * 16)>
#map1 = affine_map<(d0) -> (d0 * 16 + 16)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0) -> (d0 + 1)>

func.func @tiled(%arg0: memref<*xf32>) {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x224x224x64xf32>
  %cast = memref.cast %arg0 : memref<*xf32> to memref<64xf32>
  affine.for %arg1 = 0 to 4 {
    affine.for %arg2 = 0 to 224 {
      affine.for %arg3 = 0 to 14 {
        %alloc_0 = memref.alloc() : memref<1x16x1x16xf32>
        affine.for %arg4 = #map(%arg1) to #map1(%arg1) {
          affine.for %arg5 = #map(%arg3) to #map1(%arg3) {
            %0 = affine.load %cast[%arg4] : memref<64xf32>
            // expected-remark@above {{contiguous along loop 3}}
            affine.store %0, %alloc_0[0, %arg1 * -16 + %arg4, 0, %arg3 * -16 + %arg5] : memref<1x16x1x16xf32>
            // expected-remark@above {{contiguous along loop 4}}
            // expected-remark@above {{contiguous along loop 2}}
          }
        }
        affine.for %arg4 = #map(%arg1) to #map1(%arg1) {
          affine.for %arg5 = #map2(%arg2) to #map3(%arg2) {
            affine.for %arg6 = #map(%arg3) to #map1(%arg3) {
              %0 = affine.load %alloc_0[0, %arg1 * -16 + %arg4, -%arg2 + %arg5, %arg3 * -16 + %arg6] : memref<1x16x1x16xf32>
              // expected-remark@above {{contiguous along loop 5}}
              // expected-remark@above {{contiguous along loop 2}}
              affine.store %0, %alloc[0, %arg5, %arg6, %arg4] : memref<1x224x224x64xf32>
              // expected-remark@above {{contiguous along loop 3}}
            }
          }
        }
        memref.dealloc %alloc_0 : memref<1x16x1x16xf32>
      }
    }
  }
  return
}
