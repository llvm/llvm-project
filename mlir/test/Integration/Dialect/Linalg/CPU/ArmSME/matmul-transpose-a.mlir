// RUN: mlir-opt %s \
// RUN:   -transform-interpreter -test-transform-dialect-erase-schedule \
// RUN:   -one-shot-bufferize="bufferize-function-boundaries" -canonicalize \
// RUN:   -enable-arm-streaming="mode=locally enable-za" \
// RUN:   -convert-vector-to-arm-sme -convert-arm-sme-to-scf \
// RUN:   -convert-vector-to-scf -cse -arm-sve-legalize-vector-storage \
// RUN:   -convert-vector-to-llvm=enable-arm-sme \
// RUN:   -convert-vector-to-llvm=enable-arm-sve \
// RUN:   -cse -canonicalize -allocate-arm-sme-tiles -test-lower-to-llvm | \
// RUN: %mcr_aarch64_cmd \
// RUN:   -e=main -entry-point-result=void \
// RUN:   -march=aarch64 -mattr="+sve,+sme" \
// RUN:   -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils | \
// RUN: FileCheck %s

func.func @matmul_transpose_a(%A : tensor<?x?xf32>, %B : tensor<?x?xf32>, %C : tensor<?x?xf32>) {
  %res = linalg.matmul_transpose_a ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
                                   outs(%C: tensor<?x?xf32>) -> tensor<?x?xf32>
  %xf = tensor.cast %res : tensor<?x?xf32> to tensor<*xf32>
  call @printMemrefF32(%xf) : (tensor<*xf32>) -> ()
  return
}

func.func @main() {
  %c0 = arith.constant 0 : i32
  %c4 = arith.constant 4 : index

  %A = arith.constant dense<[
    [  1.0,  2.0,  3.0,  4.0 ],
    [  5.0,  6.0,  7.0,  8.0 ],
    [  9.0, 10.0, 11.0, 12.0 ],
    [ 13.0, 14.0, 15.0, 16.0 ]
  ]> : tensor<4x4xf32>

  %A_dyn = tensor.cast %A : tensor<4x4xf32> to tensor<?x?xf32>

  %C_init = bufferization.alloc_tensor(%c4, %c4) : tensor<?x?xf32>
  %C = linalg.fill ins(%c0 : i32) outs(%C_init : tensor<?x?xf32>) -> tensor<?x?xf32>

  // CHECK: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
  // CHECK: [276, 304, 332, 360]
  // CHECK: [304, 336, 368, 400]
  // CHECK: [332, 368, 404, 440]
  // CHECK: [360, 400, 440, 480]
  call @matmul_transpose_a(%A_dyn, %A_dyn, %C) : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) -> ()

  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module : !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul_transpose_a"]} in %module
      : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loops:3 = transform.structured.tile_using_for %0[[4], [4], 1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %tiled_linalg_op vector_sizes [[4], [4], 1]
      : !transform.any_op

    %func = transform.structured.match ops{["func.func"]} in %module
      : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %func {
      transform.apply_patterns.vector.lower_masked_transfers
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
    } : !transform.any_op

    transform.apply_patterns to %func {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
      transform.apply_patterns.vector.lower_masks
    } : !transform.any_op

    transform.yield
  }
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
