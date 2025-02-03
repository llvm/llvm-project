// DEFINE: %{compile} =  mlir-opt %s \
// DEFINE:    -transform-interpreter -test-transform-dialect-erase-schedule \
// DEFINE:    -one-shot-bufferize="bufferize-function-boundaries" \
// DEFINE:    -buffer-deallocation-pipeline="private-function-dynamic-ownership" \
// DEFINE:    -cse -canonicalize -test-lower-to-llvm
// DEFINE: %{entry_point} = main
// DEFINE: %{run} = mlir-runner -e %{entry_point} -entry-point-result=void \
// DEFINE:    -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils

// RUN: %{compile} | %{run} | FileCheck %s

/// End-to-end test for computing matrix-multiplication using linalg.mmt4d. In
/// particular, demonstrates how the following MLIR sequence (implemented in @mmt4d):
///
///   A_pack = tensor.pack A
///   B_pack = tensor.pack B
///   C_pack = tensor.pack C
///   out_pack = linalg.mmt4d(A_pack, B_pack, C_pack)
///
/// is equivalent to:
///
///  linalg.matmul(A, B, C)
///
/// (implemented in @matmul).

func.func @main() {
  // Allocate and initialise the inputs
  %A_alloc = tensor.empty() : tensor<7x16xi32>
  %B_alloc = tensor.empty() : tensor<16x13xi32>

  %three = arith.constant 3 : i32
  %four = arith.constant 4 : i32
  %A = linalg.fill ins(%three : i32) outs(%A_alloc : tensor<7x16xi32>) -> tensor<7x16xi32>
  %B = linalg.fill ins(%four : i32) outs(%B_alloc : tensor<16x13xi32>) -> tensor<16x13xi32>
  %C = arith.constant dense<[
    [ 1,  8, 15, 22, 29, 36, 43, 50, 57, 64, 71, 78, 85],
    [ 2,  9, 16, 23, 30, 37, 44, 51, 58, 65, 72, 79, 86],
    [ 3, 10, 17, 24, 31, 38, 45, 52, 59, 66, 73, 80, 87],
    [ 4, 11, 18, 25, 32, 39, 46, 53, 60, 67, 74, 81, 88],
    [ 5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89],
    [ 6, 13, 20, 27, 34, 41, 48, 55, 62, 69, 76, 83, 90],
    [ 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91]
  ]> : tensor<7x13xi32>
  
  // Matrix multiplication via linalg.mmt4d
  // CHECK: Unranked Memref
  // CHECK:  [193,   200,   207,   214,   221,   228,   235,   242,   249,   256,   263,   270,   277]
  // CHECK:  [194,   201,   208,   215,   222,   229,   236,   243,   250,   257,   264,   271,   278]
  // CHECK:  [195,   202,   209,   216,   223,   230,   237,   244,   251,   258,   265,   272,   279]
  // CHECK:  [196,   203,   210,   217,   224,   231,   238,   245,   252,   259,   266,   273,   280]
  // CHECK:  [197,   204,   211,   218,   225,   232,   239,   246,   253,   260,   267,   274,   281]
  // CHECK:  [198,   205,   212,   219,   226,   233,   240,   247,   254,   261,   268,   275,   282]
  // CHECK:  [199,   206,   213,   220,   227,   234,   241,   248,   255,   262,   269,   276,   283]
  %C_mmt4d = func.call @mmt4d(%A, %B, %C) : (tensor<7x16xi32>, tensor<16x13xi32>, tensor<7x13xi32>) -> tensor<7x13xi32>
  %xf = tensor.cast %C_mmt4d : tensor<7x13xi32> to tensor<*xi32>
  call @printMemrefI32(%xf) : (tensor<*xi32>) -> ()

  // Matrix multiplication with linalg.matmul
  // CHECK: Unranked Memref
  // CHECK:  [193,   200,   207,   214,   221,   228,   235,   242,   249,   256,   263,   270,   277]
  // CHECK:  [194,   201,   208,   215,   222,   229,   236,   243,   250,   257,   264,   271,   278]
  // CHECK:  [195,   202,   209,   216,   223,   230,   237,   244,   251,   258,   265,   272,   279]
  // CHECK:  [196,   203,   210,   217,   224,   231,   238,   245,   252,   259,   266,   273,   280]
  // CHECK:  [197,   204,   211,   218,   225,   232,   239,   246,   253,   260,   267,   274,   281]
  // CHECK:  [198,   205,   212,   219,   226,   233,   240,   247,   254,   261,   268,   275,   282]
  // CHECK:  [199,   206,   213,   220,   227,   234,   241,   248,   255,   262,   269,   276,   283]
  %C_matmul = func.call @matmul(%A, %B, %C) : (tensor<7x16xi32>, tensor<16x13xi32>, tensor<7x13xi32>) -> tensor<7x13xi32>
  %xf_2 = tensor.cast %C_matmul : tensor<7x13xi32> to tensor<*xi32>
  call @printMemrefI32(%xf_2) : (tensor<*xi32>) -> ()

  return
}

func.func private @matmul(%A: tensor<7x16xi32>, %B: tensor<16x13xi32>, %C: tensor<7x13xi32>) -> tensor<7x13xi32> {
  %C_matmul = linalg.matmul ins(%A, %B: tensor<7x16xi32>, tensor<16x13xi32>) 
                            outs(%C: tensor<7x13xi32>) -> tensor<7x13xi32>

  return %C_matmul : tensor<7x13xi32>
}

func.func private @mmt4d(%A: tensor<7x16xi32>, %B: tensor<16x13xi32>, %C: tensor<7x13xi32>) -> tensor<7x13xi32> {
  %zero = arith.constant 0 : i32

  %A_pack_empty = tensor.empty() : tensor<2x16x8x1xi32>
  %B_pack_empty = tensor.empty() : tensor<2x16x8x1xi32>
  %C_pack_empty = tensor.empty() : tensor<2x2x8x8xi32>

  // Pack matrices
  %A_pack = tensor.pack %A padding_value(%zero : i32) inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %A_pack_empty : tensor<7x16xi32> -> tensor<2x16x8x1xi32>
  %B_pack = tensor.pack %B padding_value(%zero : i32) outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [8, 1] into %B_pack_empty : tensor<16x13xi32> -> tensor<2x16x8x1xi32>
  %C_pack = tensor.pack %C padding_value(%zero : i32) outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %C_pack_empty : tensor<7x13xi32> -> tensor<2x2x8x8xi32>

  // MMT4D
  %mmt4d = linalg.mmt4d ins(%A_pack, %B_pack : tensor<2x16x8x1xi32>, tensor<2x16x8x1xi32>) outs(%C_pack : tensor<2x2x8x8xi32>) -> tensor<2x2x8x8xi32>

  // Unpack output
  %C_out_empty = tensor.empty() : tensor<7x13xi32>
  %C_out_unpack = tensor.unpack %mmt4d outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %C_out_empty : tensor<2x2x8x8xi32> -> tensor<7x13xi32>

  return %C_out_unpack : tensor<7x13xi32>
}

module @transforms attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
   %mmt4d = transform.collect_matching @match_mmt4d in %module : (!transform.any_op) -> (!transform.any_op)
   %func = transform.get_parent_op %mmt4d {isolated_from_above} : (!transform.any_op) -> !transform.op<"func.func">

   // Step 1: Tile
   // Tile parallel dims
   %tiled_linalg_op_p, %loops:4 = transform.structured.tile_using_for %mmt4d tile_sizes [1, 1, 0, 8, 8, 0]
     : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
   // Tile reduction dims
   %tiled_linalg_op_r, %loops2:2 = transform.structured.tile_using_for %tiled_linalg_op_p tile_sizes [0, 0, 1, 0, 0, 1]
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

   // Step 4. Lower tensor.pack
   %pack = transform.structured.match ops{["tensor.pack"]} in %func_h
     : (!transform.op<"func.func">) -> !transform.op<"tensor.pack">
   transform.structured.lower_pack %pack : (!transform.op<"tensor.pack">)
     -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)

   // Step 5. Lower tensor.unpack
   %unpack = transform.structured.match ops{["tensor.unpack"]} in %func_h
      : (!transform.op<"func.func">) -> !transform.op<"tensor.unpack">
    transform.structured.lower_unpack %unpack : (!transform.op<"tensor.unpack">)
      -> (!transform.op<"tensor.empty">,
          !transform.op<"linalg.transpose">,
          !transform.op<"tensor.collapse_shape">,
          !transform.op<"tensor.extract_slice">)
    transform.yield
  }

  transform.named_sequence @match_mmt4d(
      %entry: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %entry ["linalg.mmt4d"] : !transform.any_op
    transform.yield %entry : !transform.any_op
  }
}

func.func private @printMemrefI32(%ptr : tensor<*xi32>)
