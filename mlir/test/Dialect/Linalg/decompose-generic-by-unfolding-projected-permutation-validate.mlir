// RUN: mlir-opt %s -linalg-specialize-generic-ops -verify-diagnostics

// Fixes issue: 122094. Verify that the following code compiles without issue.

func.func @test_broadcast_scalar_across_single_tensor() -> tensor<2x2xi32> {

  %a = arith.constant dense<2> : tensor<2x2xi32>
  %b = arith.constant 42 : i32
  %c = tensor.empty() : tensor<2x2xi32>
  %res = linalg.generic
    {
      indexing_maps = [
        affine_map<(i, j) -> (i, j)>, 
        affine_map<(i, j) -> ()>,     
        affine_map<(i, j) -> (i, j)>  
      ],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%a, %b : tensor<2x2xi32>, i32)
    outs(%c : tensor<2x2xi32>) {
  ^bb0(%x: i32, %scalar: i32, %out: i32):
    %sum = arith.addi %x, %scalar : i32
    linalg.yield %sum : i32
  } -> tensor<2x2xi32>

  return %res : tensor<2x2xi32>
}
