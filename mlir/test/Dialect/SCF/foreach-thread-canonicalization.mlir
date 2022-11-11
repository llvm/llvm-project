// RUN: mlir-opt %s -scf-for-loop-canonicalization | FileCheck %s

func.func @reduce() {
  // CHECK: %[[C64:.*]] = arith.constant 64 : index
  %c2 = arith.constant 2 : index
  %cst_0 = arith.constant -0.000000e+00 : f32
  %0 = memref.alloc() : memref<128x384xf32>
  linalg.fill ins(%cst_0 : f32) outs(%0 : memref<128x384xf32>)
  %2 = memref.alloc() : memref<128xf32>
  linalg.fill ins(%cst_0 : f32) outs(%2 : memref<128xf32>)
  scf.foreach_thread (%arg0) in (%c2) {
    %7 = affine.min affine_map<(d0) -> (d0 * -64 + 128, 64)>(%arg0)
    %8 = affine.max affine_map<(d0) -> (0, d0)>(%7)
    %9 = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg0)
    %10 = affine.min affine_map<(d0, d1) -> (d1 * -64 + 128, d0)>(%8, %arg0)

    // CHECK: memref.subview %{{.*}}[%{{.*}}, 0] [%[[C64]], 384] [1, 1] : memref<128x384xf32> to memref<?x384xf32, {{.*}}>
    // CHECK: memref.subview %{{.*}}[%{{.*}}] [%[[C64]]] [1] : memref<128xf32> to memref<?xf32, {{.*}}>
    %11 = memref.subview %0[%9, 0] [%10, 384] [1, 1] : 
      memref<128x384xf32> to memref<?x384xf32, affine_map<(d0, d1)[s0] -> (d0 * 384 + s0 + d1)>>
    %12 = memref.subview %2[%9] [%10] [1] : 
      memref<128xf32> to memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>>

    // CHECK: linalg.generic {{.*}} ins(%{{.*}} : memref<?x384xf32, {{.*}}>) outs(%{{.*}} : memref<?xf32, {{.*}}>)
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, 
                                      affine_map<(d0, d1) -> (d0)>], 
                     iterator_types = ["parallel", "reduction"]} 
      ins(%11 : memref<?x384xf32, affine_map<(d0, d1)[s0] -> (d0 * 384 + s0 + d1)>>) 
      outs(%12 : memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>>) {
        ^bb0(%arg1: f32, %arg2: f32):
          %14 = arith.addf %arg1, %arg2 : f32
          linalg.yield %14 : f32
      }
  }
  return
}
