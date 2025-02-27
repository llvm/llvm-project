// RUN: mlir-opt %s \
// RUN:   -transform-interpreter -test-transform-dialect-erase-schedule \
// RUN:   -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:   -test-lower-to-arm-sme -test-lower-to-llvm | \
// RUN: %mcr_aarch64_cmd \
// RUN:   -e=main -entry-point-result=void \
// RUN:   -march=aarch64 -mattr="+sve,+sme" \
// RUN:   -shared-libs=%native_mlir_runner_utils,%native_mlir_c_runner_utils,%native_arm_sme_abi_shlib | \
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
  %c7 = arith.constant 7 : index

  %A = arith.constant dense<[
    [ 1.,  2.,  3.,  4.,  5.,  6.,  7.],
    [ 8.,  9., 10., 11., 12., 13., 14.],
    [15., 16., 17., 18., 19., 20., 21.],
    [22., 23., 24., 25., 26., 27., 28.],
    [29., 30., 31., 32., 33., 34., 35.],
    [36., 37., 38., 39., 40., 41., 42.],
    [43., 44., 45., 46., 47., 48., 49.],
    [50., 51., 52., 53., 54., 55., 56.],
    [57., 58., 59., 60., 61., 62., 63.],
    [64., 65., 66., 67., 68., 69., 70.],
    [71., 72., 73., 74., 75., 76., 77.],
    [78., 79., 80., 81., 82., 83., 84.],
    [85., 86., 87., 88., 89., 90., 91.]
  ]> : tensor<13x7xf32>

  %A_dyn = tensor.cast %A : tensor<13x7xf32> to tensor<?x?xf32>

  %C_init = bufferization.alloc_tensor(%c7, %c7) : tensor<?x?xf32>
  %C = linalg.fill ins(%c0 : i32) outs(%C_init : tensor<?x?xf32>) -> tensor<?x?xf32>

  // CHECK: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [7, 7] strides = [7, 1] data =
  // CHECK: [32955, 33514, 34073, 34632, 35191, 35750, 36309]
  // CHECK: [33514, 34086, 34658, 35230, 35802, 36374, 36946]
  // CHECK: [34073, 34658, 35243, 35828, 36413, 36998, 37583]
  // CHECK: [34632, 35230, 35828, 36426, 37024, 37622, 38220]
  // CHECK: [35191, 35802, 36413, 37024, 37635, 38246, 38857]
  // CHECK: [35750, 36374, 36998, 37622, 38246, 38870, 39494]
  // CHECK: [36309, 36946, 37583, 38220, 38857, 39494, 40131]
  call @matmul_transpose_a(%A_dyn, %A_dyn, %C) : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) -> ()

  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module : !transform.any_op {transform.readonly}) {
    %matmul_transpose_a = transform.structured.match ops{["linalg.matmul_transpose_a"]} in %module
      : (!transform.any_op) -> !transform.any_op

    // Step 1: Tile for size [4] x [4], which corresponds to SVLs x SVLs, where
    //         SVLs is the number of 32-bit elements in a vector of SVL bits.
    %tiled_linalg_op, %loops:3 = transform.structured.tile_using_for %matmul_transpose_a tile_sizes [[4], [4], 1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // Step 2: Vectorize.
    transform.structured.vectorize %tiled_linalg_op vector_sizes [[4], [4], 1]
      : !transform.any_op

    %func = transform.structured.match ops{["func.func"]} in %module
      : (!transform.any_op) -> !transform.any_op

    // Step 3: Lower vector.multi_reduction to vector.contract (+ some helpful patterns).
    transform.apply_patterns to %func {
      transform.apply_patterns.vector.lower_masked_transfers
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
    } : !transform.any_op

    // Step 4: Lower vector.contract to vector.outerproduct.
    transform.apply_patterns to %func {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
      transform.apply_patterns.vector.lower_masks
      transform.apply_patterns.canonicalization
    } : !transform.any_op

    // Step 5 (optional optimization): Hoist accumulator load/store.
    %func_h = transform.structured.hoist_redundant_vector_transfers %func
        : (!transform.any_op) -> !transform.any_op
    %all_loops = transform.structured.match interface{LoopLikeInterface} in %module
      : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %all_loops : !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %all_loops : !transform.any_op
    transform.yield
  }
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
