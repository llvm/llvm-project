// RUN: mlir-opt %s --convert-arith-fp8-extf-to-lut | FileCheck %s

// linalg.generic over tensor<1024xf8E4M3FN>: both extf ops inside the body
// should be replaced by LUT-lookup sequences; one shared table is emitted.

// CHECK-DAG: memref.global "private" constant @__extf_lut_f8E4M3FN : memref<256xf32>

func.func @linalg_addf_fp8(
    %a: tensor<1024xf8E4M3FN>,
    %b: tensor<1024xf8E4M3FN>) -> tensor<1024xf32> {
  // CHECK-LABEL: @linalg_addf_fp8
  // CHECK:         linalg.generic
  // CHECK-COUNT-2: memref.get_global @__extf_lut_f8E4M3FN : memref<256xf32>
  // CHECK:         arith.addf {{.*}} : f32
  // CHECK-NOT:     arith.extf
  %init = tensor.empty() : tensor<1024xf32>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%a, %b : tensor<1024xf8E4M3FN>, tensor<1024xf8E4M3FN>)
    outs(%init : tensor<1024xf32>) {
  ^bb0(%in_a: f8E4M3FN, %in_b: f8E4M3FN, %out: f32):
    %ra = arith.extf %in_a : f8E4M3FN to f32
    %rb = arith.extf %in_b : f8E4M3FN to f32
    %sum = arith.addf %ra, %rb : f32
    linalg.yield %sum : f32
  } -> tensor<1024xf32>
  return %result : tensor<1024xf32>
}
