// RUN: mlir-opt %s -scf-for-loop-canonicalization -canonicalize | FileCheck %s

func.func @reduce() -> tensor<128xf32> {
  %c2 = arith.constant 2 : index
  %cst = arith.constant dense<1.000000e+00> : tensor<1x128x384xf32>
  %cst_0 = arith.constant -0.000000e+00 : f32
  %0 = linalg.init_tensor [128, 384] : tensor<128x384xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<128x384xf32>) -> tensor<128x384xf32>
  %2 = linalg.init_tensor [128] : tensor<128xf32>
  %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<128xf32>) -> tensor<128xf32>
  %4 = scf.foreach_thread (%arg0) in (%c2) -> (tensor<128xf32>) {
    %7 = affine.min affine_map<(d0) -> (d0 * -64 + 128, 64)>(%arg0)
    %8 = affine.max affine_map<(d0) -> (0, d0)>(%7)
    %9 = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg0)
    %10 = affine.min affine_map<(d0, d1) -> (d1 * -64 + 128, d0)>(%8, %arg0)

    // CHECK: tensor.extract_slice %{{.*}}[%{{.*}}, 0] [64, 384] [1, 1] : tensor<128x384xf32> to tensor<64x384xf32>
    // CHECK: tensor.extract_slice %{{.*}}[%{{.*}}] [64] [1] : tensor<128xf32> to tensor<64xf32>
    %11 = tensor.extract_slice %1[%9, 0] [%10, 384] [1, 1] : tensor<128x384xf32> to tensor<?x384xf32>
    %12 = tensor.extract_slice %3[%9] [%10] [1] : tensor<128xf32> to tensor<?xf32>

    // CHECK: linalg.generic {{.*}} ins(%{{.*}} : tensor<64x384xf32>) outs(%{{.*}} : tensor<64xf32>) {
    %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%11 : tensor<?x384xf32>) outs(%12 : tensor<?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %14 = arith.addf %arg1, %arg2 : f32
      linalg.yield %14 : f32
    } -> tensor<?xf32>

    // TODO: canonicalize this cast away.
    // CHECK: %[[dyn_casted:.*]] = tensor.cast %{{.*}} : tensor<64xf32> to tensor<?xf32>
    // CHECK: scf.foreach_thread.parallel_insert_slice %[[dyn_casted:.*]] into %{{.*}}[%{{.*}}] [64] [1] : tensor<?xf32> into tensor<128xf32>
    scf.foreach_thread.perform_concurrently {
      scf.foreach_thread.parallel_insert_slice %13 into %3[%9] [%10] [1] : tensor<?xf32> into tensor<128xf32>
    }
  }
  return %4 : tensor<128xf32>
}
