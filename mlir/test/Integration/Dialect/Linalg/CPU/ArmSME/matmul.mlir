// RUN: mlir-opt %s \
// RUN:   -transform-interpreter -test-transform-dialect-erase-schedule \
// RUN:   -canonicalize \
// RUN:   -convert-vector-to-arm-sme -allocate-arm-sme-tiles -convert-arm-sme-to-scf \
// RUN:   -enable-arm-streaming="streaming-mode=streaming-locally za-mode=new-za only-if-required-by-ops" \
// RUN:   -convert-vector-to-scf -cse -arm-sve-legalize-vector-storage \
// RUN:   -convert-arm-sme-to-llvm \
// RUN:   -convert-vector-to-llvm=enable-arm-sve \
// RUN:   -cse -canonicalize -test-lower-to-llvm | \
// RUN: %mcr_aarch64_cmd \
// RUN:   -e=main -entry-point-result=void \
// RUN:   -march=aarch64 -mattr="+sve,+sme" \
// RUN:   -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%arm_sme_abi_shlib | \
// RUN: FileCheck %s

func.func @matmul(%A : tensor<?x?xf32>, %B : tensor<?x?xf32>, %C : tensor<?x?xf32>) {
  %res = linalg.matmul ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
                       outs(%C: tensor<?x?xf32>) -> tensor<?x?xf32>
  %xf = tensor.cast %res : tensor<?x?xf32> to tensor<*xf32>
  call @printMemrefF32(%xf) : (tensor<*xf32>) -> ()
  return
}

func.func @main() {
  %c0 = arith.constant 0 : i32
  %c7 = arith.constant 7 : index

  %A = arith.constant dense<[
    [ 1.,  8., 15., 22., 29., 36., 43., 50., 57., 64., 71., 78., 85.],
    [ 2.,  9., 16., 23., 30., 37., 44., 51., 58., 65., 72., 79., 86.],
    [ 3., 10., 17., 24., 31., 38., 45., 52., 59., 66., 73., 80., 87.],
    [ 4., 11., 18., 25., 32., 39., 46., 53., 60., 67., 74., 81., 88.],
    [ 5., 12., 19., 26., 33., 40., 47., 54., 61., 68., 75., 82., 89.],
    [ 6., 13., 20., 27., 34., 41., 48., 55., 62., 69., 76., 83., 90.],
    [ 7., 14., 21., 28., 35., 42., 49., 56., 63., 70., 77., 84., 91.]
  ]> : tensor<7x13xf32>

  %B_init = tensor.empty() : tensor<13x7xf32>
  %B = linalg.transpose ins(%A: tensor<7x13xf32>)
                        outs(%B_init: tensor<13x7xf32>) permutation = [1, 0]

  %A_dyn = tensor.cast %A : tensor<7x13xf32> to tensor<?x?xf32>
  %B_dyn = tensor.cast %B : tensor<13x7xf32> to tensor<?x?xf32>

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
  call @matmul(%A_dyn, %B_dyn, %C) : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) -> ()

  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module : !transform.any_op {transform.consumed}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %module
      : (!transform.any_op) -> !transform.any_op

    // Step 1: Tile for size [4] x [4], which corresponds to SVLs x SVLs, where
    // SVLs is the number of 32-bit elements in a vector of SVL bits.
    %tiled_linalg_op, %loops:3 = transform.structured.tile_using_for %matmul[[4], [4], 1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // Step 2: Vectorize.
    transform.structured.vectorize %tiled_linalg_op vector_sizes [[4], [4], 1]
      : !transform.any_op

    // Step 3: Bufferize ahead of TransferReadDropUnitDimsPattern, which
    // currently only supports memrefs.
    %bufferize = transform.bufferization.one_shot_bufferize %module
      {bufferize_function_boundaries=true} : (!transform.any_op) -> !transform.any_op

    %func = transform.structured.match ops{["func.func"]} in %bufferize
      : (!transform.any_op) -> !transform.any_op

    // Step 4: Lower vector.multi_reduction to vector.contract (+ some helpful patterns).
    transform.apply_patterns to %func {
      transform.apply_patterns.vector.lower_masked_transfers
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
    } : !transform.any_op

    // Step 5: Lower vector.contract to vector.outerproduct. Also drop unit
    // dims, specifically to prevent vector.transfer_read of vector<[4]x1xf32>,
    // which can't be lowered in generic path.
    transform.apply_patterns to %func {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
      transform.apply_patterns.vector.lower_masks
      transform.apply_patterns.vector.rank_reducing_subview_patterns
    } : !transform.any_op

    transform.yield
  }
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
