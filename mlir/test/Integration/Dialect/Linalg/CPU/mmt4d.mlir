// DEFINE: %{compile} =  mlir-opt %s \
// DEFINE:    -transform-interpreter -test-transform-dialect-erase-schedule \
// DEFINE:    -one-shot-bufferize -func-bufferize -cse -canonicalize -convert-vector-to-scf -test-lower-to-llvm -o %t
// DEFINE: %{entry_point} = mmt4d
// DEFINE: %{run} = mlir-cpu-runner %t -e %{entry_point} -entry-point-result=void \
// DEFINE:    -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils

// RUN: %{compile}

// RUN: %{run} | FileCheck %s

func.func @mmt4d() {
  // Allocate the matrices
  %A_alloc = tensor.empty() : tensor<2x2x3x1xi32>
  %B_alloc = tensor.empty() : tensor<2x2x3x1xi32>
  %C_alloc = tensor.empty() : tensor<2x2x3x3xi32>
  %C_in = arith.constant dense<[
    [[[ 1, 2, 3],
     [ 4, 5, 6],
     [ 7, 8, 9]],
    [[ 11, 12, 13],
     [ 14, 15, 16],
     [ 17, 18, 19]]],
    [[[ 21, 22, 23],
     [ 24, 25, 26],
     [ 27, 28, 29]],
    [[ 31, 32, 33],
     [ 34, 35, 36],
     [ 37, 38, 39]]]
  ]> : tensor<2x2x3x3xi32>

  // Initialise the matrices
  %three = arith.constant 3 : i32
  %four = arith.constant 4 : i32
  %A = linalg.fill ins(%three : i32) outs(%A_alloc : tensor<2x2x3x1xi32>) -> tensor<2x2x3x1xi32>
  %B = linalg.fill ins(%four : i32) outs(%B_alloc : tensor<2x2x3x1xi32>) -> tensor<2x2x3x1xi32>

  // Matmul
  %C_out = linalg.mmt4d ins(%A, %B: tensor<2x2x3x1xi32>, tensor<2x2x3x1xi32>) outs(%C_in: tensor<2x2x3x3xi32>) -> tensor<2x2x3x3xi32>

  // Print and verify the output
  // CHECK:  Unranked Memref {{.*}} rank = 4 offset = 0 sizes = [2, 2, 3, 3] strides = [18, 9, 3, 1] data =
  // C[0, 0]
  // CHECK-NEXT: [25,  26, 27]
  // CHECK-NEXT: [28,  29, 30]
  // CHECK-NEXT: [31,  32, 33]
  // C[0, 1]
  // CHECK-NEXT: [35,  36, 37]
  // CHECK-NEXT: [38,  39, 40]
  // CHECK-NEXT: [41,  42, 43]
  // C[1, 0]
  // CHECK-NEXT: [45,  46, 47]
  // CHECK-NEXT: [48,  49, 50]
  // CHECK-NEXT: [51,  52, 53]
  // C[1, 1]
  // CHECK-NEXT: [55,  56, 57]
  // CHECK-NEXT: [58,  59, 60]
  // CHECK-NEXT: [61,  62, 63]

  %xf = tensor.cast %C_out : tensor<2x2x3x3xi32> to tensor<*xi32>
  call @printMemrefI32(%xf) : (tensor<*xi32>) -> ()

  return
}

module @transforms attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
   %mmt4d = transform.collect_matching @match_mmt4d in %module : (!transform.any_op) -> (!transform.any_op)
   %func = transform.get_parent_op %mmt4d {isolated_from_above} : (!transform.any_op) -> !transform.op<"func.func">

   // Step 1: Tile
   // Tile parallel dims
   %tiled_linalg_op_p, %loops:4 = transform.structured.tile_using_for %mmt4d[1, 1, 0, 3, 3, 0]
     : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
   // Tile reduction dims
   %tiled_linalg_op_r, %loops2:2 = transform.structured.tile_using_for %tiled_linalg_op_p[0, 0, 1, 0, 0, 1]
     : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

   // Step 2: Vectorize
   transform.structured.vectorize %tiled_linalg_op_r : !transform.any_op

   // Step 3: Simplify
   // vector.multi_reduction --> vector.contract
   // Generates a 6-dim vector.contract with the dim matching the original MMT4D Op
   // and with the following split into parallel and reduction dims:
   //    * parallel, parallel, reduction, parallel, parallel, reduction
   transform.apply_patterns to %func {
     transform.apply_patterns.vector.reduction_to_contract
     // Reduce the rank of xfer ops. This transforms vector.contract to be
     // more matmul-like and to enable the lowering to outer product Ops.
     transform.apply_patterns.vector.transfer_permutation_patterns
   } : !transform.op<"func.func">

   // Hoisting and LICM - not strictly required
   %func_h = transform.structured.hoist_redundant_vector_transfers %func
     : (!transform.op<"func.func">) -> !transform.op<"func.func">
   %all_loops = transform.structured.match interface{LoopLikeInterface} in %func_h
     : (!transform.op<"func.func">) -> !transform.any_op
   transform.apply_licm to %all_loops : !transform.any_op
   transform.loop.hoist_loop_invariant_subsets %all_loops : !transform.any_op

   // Simplify the 6-dim vector.contract into a 3-dim matmul-like
   // vector.contract with the following split into parallel and reduction
   // dims:
   //    * parallel, parallel, reduction
   transform.apply_patterns to %func_h {
     transform.apply_patterns.vector.reduction_to_contract
     transform.apply_patterns.vector.cast_away_vector_leading_one_dim
     transform.apply_patterns.canonicalization
   } : !transform.op<"func.func">
    transform.yield
  }

  transform.named_sequence @match_mmt4d(
      %entry: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %entry ["linalg.mmt4d"] : !transform.any_op
    transform.yield %entry : !transform.any_op
  }
}

func.func private @printMemrefI32(%ptr : tensor<*xi32>)
