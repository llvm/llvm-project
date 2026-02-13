// RUN: mlir-opt %s --sparsification-and-bufferization        | FileCheck %s --check-prefix=CHECK-NOVEC
// RUN: mlir-opt %s --sparsification-and-bufferization="vl=8" | FileCheck %s --check-prefix=CHECK-VEC

// Test to ensure we can pass optimization flags into
// the mini sparsification and bufferization pipeline.

#SV = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

#trait_sum_reduction = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a
    affine_map<(i) -> ()>    // x (scalar out)
  ],
  iterator_types = ["reduction"],
  doc = "x += SUM_i a(i)"
}

//
// CHECK-NOVEC-LABEL: func.func @sum_reduction
// CHECK-NOVEC:       scf.for
// CHECK-NOVEC:         arith.addf %{{.*}} %{{.*}} : f32
// CHECK-NOVEC:       }
//
// CHECK-VEC-LABEL: func.func @sum_reduction
// CHECK-VEC:       vector.insert
// CHECK-VEC:       scf.for
// CHECK-VEC:         vector.create_mask
// CHECK-VEC:         vector.maskedload
// CHECK-VEC:         arith.addf %{{.*}} %{{.*}} : vector<8xf32>
// CHECK-VEC:       }
// CHECK-VEC:       vector.reduction <add>
//
func.func @sum_reduction(%arga: tensor<?xf32, #SV>,
                         %argx: tensor<f32>) -> tensor<f32> {
  %0 = linalg.generic #trait_sum_reduction
    ins(%arga: tensor<?xf32, #SV>)
    outs(%argx: tensor<f32>) {
      ^bb(%a: f32, %x: f32):
        %0 = arith.addf %x, %a : f32
        linalg.yield %0 : f32
  } -> tensor<f32>
  return %0 : tensor<f32>
}
