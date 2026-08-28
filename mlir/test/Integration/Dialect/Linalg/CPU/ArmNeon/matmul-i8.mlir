// DEFINE: %{compile} = mlir-opt %s \
// DEFINE:   -transform-interpreter -test-transform-dialect-erase-schedule \
// DEFINE:   -cse -canonicalize -convert-vector-to-scf \
// DEFINE:   -convert-vector-to-llvm="enable-arm-neon enable-arm-i8mm" -test-lower-to-llvm \
// DEFINE: -o %t

// DEFINE: %{run} = %mcr_aarch64_cmd %t -e main -entry-point-result=void --march=aarch64 --mattr="+neon,+i8mm" \
// DEFINE:    -shared-libs=%native_mlir_runner_utils,%native_mlir_c_runner_utils

// RUN: rm -f %t && %{compile} && FileCheck %s --input-file=%t -check-prefix CHECK-IR && %{run} | FileCheck %s

// End-to-end test for `linalg.matmul` on i8 operands accumulating to i32,
// lowered via `linalg.pack -> linalg.mmt4d -> linalg.unpack` down to Arm's
// FEAT_I8MM `smmla`. Packing gives the inner tiles a statically-known shape,
// so the vectorized `vector.contract` never needs masking, and
// `linalg.mmt4d`'s RHS is already N-major (that's the "t" in "mmt4d"),
// exactly what `LowerContractionToNeonI8MMPattern` expects -- no
// transpose_matmul step needed here, unlike a plain `linalg.matmul`.

func.func @main() {
  %A_empty = tensor.empty() : tensor<7x16xi8>
  %B_empty = tensor.empty() : tensor<16x13xi8>

  %c3 = arith.constant 3 : i8
  %c4 = arith.constant 4 : i8
  %A = linalg.fill ins(%c3 : i8) outs(%A_empty : tensor<7x16xi8>) -> tensor<7x16xi8>
  %B = linalg.fill ins(%c4 : i8) outs(%B_empty : tensor<16x13xi8>) -> tensor<16x13xi8>
  %C = arith.constant dense<[
    [ 1,  8, 15, 22, 29, 36, 43, 50, 57, 64, 71, 78, 85],
    [ 2,  9, 16, 23, 30, 37, 44, 51, 58, 65, 72, 79, 86],
    [ 3, 10, 17, 24, 31, 38, 45, 52, 59, 66, 73, 80, 87],
    [ 4, 11, 18, 25, 32, 39, 46, 53, 60, 67, 74, 81, 88],
    [ 5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89],
    [ 6, 13, 20, 27, 34, 41, 48, 55, 62, 69, 76, 83, 90],
    [ 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91]
  ]> : tensor<7x13xi32>

  // VARIANT: Matrix multiplication via linalg.mmt4d
  // CHECK: Unranked Memref
  // CHECK:  [193,   200,   207,   214,   221,   228,   235,   242,   249,   256,   263,   270,   277]
  // CHECK:  [194,   201,   208,   215,   222,   229,   236,   243,   250,   257,   264,   271,   278]
  // CHECK:  [195,   202,   209,   216,   223,   230,   237,   244,   251,   258,   265,   272,   279]
  // CHECK:  [196,   203,   210,   217,   224,   231,   238,   245,   252,   259,   266,   273,   280]
  // CHECK:  [197,   204,   211,   218,   225,   232,   239,   246,   253,   260,   267,   274,   281]
  // CHECK:  [198,   205,   212,   219,   226,   233,   240,   247,   254,   261,   268,   275,   282]
  // CHECK:  [199,   206,   213,   220,   227,   234,   241,   248,   255,   262,   269,   276,   283]
  %C_mmt4d = func.call @matmul_via_mmt4d(%A, %B, %C) : (tensor<7x16xi8>, tensor<16x13xi8>, tensor<7x13xi32>) -> tensor<7x13xi32>
  %C_mmt4d_cast = tensor.cast %C_mmt4d : tensor<7x13xi32> to tensor<*xi32>
  vector.print str "RESULT FROM linalg.mmt4d:\n"
  call @printMemrefI32(%C_mmt4d_cast) : (tensor<*xi32>) -> ()

  // VARIANT: Matrix multiplication via linalg.matmul (cross-check)
  // CHECK: Unranked Memref
  // CHECK:  [193,   200,   207,   214,   221,   228,   235,   242,   249,   256,   263,   270,   277]
  // CHECK:  [194,   201,   208,   215,   222,   229,   236,   243,   250,   257,   264,   271,   278]
  // CHECK:  [195,   202,   209,   216,   223,   230,   237,   244,   251,   258,   265,   272,   279]
  // CHECK:  [196,   203,   210,   217,   224,   231,   238,   245,   252,   259,   266,   273,   280]
  // CHECK:  [197,   204,   211,   218,   225,   232,   239,   246,   253,   260,   267,   274,   281]
  // CHECK:  [198,   205,   212,   219,   226,   233,   240,   247,   254,   261,   268,   275,   282]
  // CHECK:  [199,   206,   213,   220,   227,   234,   241,   248,   255,   262,   269,   276,   283]
  %C_matmul = func.call @matmul(%A, %B, %C) : (tensor<7x16xi8>, tensor<16x13xi8>, tensor<7x13xi32>) -> tensor<7x13xi32>
  %C_matmul_cast = tensor.cast %C_matmul : tensor<7x13xi32> to tensor<*xi32>
  vector.print str "RESULT FROM linalg.matmul:\n"
  call @printMemrefI32(%C_matmul_cast) : (tensor<*xi32>) -> ()

  return
}

func.func private @matmul(%A: tensor<7x16xi8>, %B: tensor<16x13xi8>, %C: tensor<7x13xi32>) -> tensor<7x13xi32> {
  %C_matmul = linalg.matmul ins(%A, %B: tensor<7x16xi8>, tensor<16x13xi8>)
                            outs(%C: tensor<7x13xi32>) -> tensor<7x13xi32>
  return %C_matmul : tensor<7x13xi32>
}

// LHS packed tile: M0=4, K0=8 (K0 must be a multiple of 8 for FEAT_I8MM).
func.func private @pack_lhs(%A: tensor<7x16xi8>) -> tensor<2x2x4x8xi8> {
  %pad = arith.constant 0 : i8
  %A_pack_empty = tensor.empty() : tensor<2x2x4x8xi8>
  %A_pack = linalg.pack %A
    padding_value(%pad : i8)
    inner_dims_pos = [0, 1]
    inner_tiles = [4, 8]
    into %A_pack_empty : tensor<7x16xi8> -> tensor<2x2x4x8xi8>
  return %A_pack : tensor<2x2x4x8xi8>
}

// RHS packed tile: N0=4, K0=8.
func.func private @pack_rhs(%B: tensor<16x13xi8>) -> tensor<4x2x4x8xi8> {
  %pad = arith.constant 0 : i8
  %B_pack_empty = tensor.empty() : tensor<4x2x4x8xi8>
  %B_pack = linalg.pack %B
    padding_value(%pad : i8)
    outer_dims_perm = [1, 0]
    inner_dims_pos = [1, 0]
    inner_tiles = [4, 8]
    into %B_pack_empty : tensor<16x13xi8> -> tensor<4x2x4x8xi8>
  return %B_pack : tensor<4x2x4x8xi8>
}

func.func private @pack_acc(%C: tensor<7x13xi32>) -> tensor<2x4x4x4xi32> {
  %pad = arith.constant 0 : i32
  %C_pack_empty = tensor.empty() : tensor<2x4x4x4xi32>
  %C_pack = linalg.pack %C
    padding_value(%pad : i32)
    outer_dims_perm = [0, 1]
    inner_dims_pos = [0, 1]
    inner_tiles = [4, 4]
    into %C_pack_empty : tensor<7x13xi32> -> tensor<2x4x4x4xi32>
  return %C_pack : tensor<2x4x4x4xi32>
}

func.func private @unpack_acc(%C_packed: tensor<2x4x4x4xi32>) -> tensor<7x13xi32> {
  %C_out_empty = tensor.empty() : tensor<7x13xi32>
  %C_out_unpack = linalg.unpack %C_packed
    outer_dims_perm = [0, 1]
    inner_dims_pos = [0, 1]
    inner_tiles = [4, 4]
    into %C_out_empty : tensor<2x4x4x4xi32> -> tensor<7x13xi32>
  return %C_out_unpack: tensor<7x13xi32>
}

// CHECK-IR-LABEL: llvm.func @matmul_via_mmt4d
// CHECK-IR-COUNT-4: arm_neon.intr.smmla
func.func private @matmul_via_mmt4d(%A: tensor<7x16xi8>, %B: tensor<16x13xi8>, %C: tensor<7x13xi32>) -> tensor<7x13xi32> {
  %A_pack = func.call @pack_lhs(%A): (tensor<7x16xi8>) -> tensor<2x2x4x8xi8>
  %B_pack = func.call @pack_rhs(%B): (tensor<16x13xi8>) -> tensor<4x2x4x8xi8>
  %C_pack = func.call @pack_acc(%C): (tensor<7x13xi32>) -> tensor<2x4x4x4xi32>

  %mmt4d = linalg.mmt4d ins(%A_pack, %B_pack : tensor<2x2x4x8xi8>, tensor<4x2x4x8xi8>) outs(%C_pack : tensor<2x4x4x4xi32>) -> tensor<2x4x4x4xi32>

  %C_out_unpack = func.call @unpack_acc(%mmt4d) : (tensor<2x4x4x4xi32>) -> tensor<7x13xi32>
  return %C_out_unpack : tensor<7x13xi32>
}

module @transforms attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.consumed}) {
    %mmt4d = transform.collect_matching @match_mmt4d in %module : (!transform.any_op) -> (!transform.any_op)
    %mmt4d_func = transform.get_parent_op %mmt4d {isolated_from_above} : (!transform.any_op) -> !transform.op<"func.func">

    // Tile parallel dims (m, n, k, m0, n0, k0): full inner tiles, one outer
    // iteration at a time.
    %tiled_mmt4d_parallel, %_:4 = transform.structured.tile_using_for %mmt4d tile_sizes [1, 1, 0, 4, 4, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    // Tile reduction dims: k0=8 is the full inner extent (FEAT_I8MM handles
    // the whole 8-wide reduction in one instruction, no further split).
    %tiled_mmt4d, %_1:2 = transform.structured.tile_using_for %tiled_mmt4d_parallel tile_sizes [0, 0, 1, 0, 0, 8]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Vectorize directly to a named `vector.contract` (compact 2-operand
    // form) instead of the generic broadcast form, since
    // LowerContractionToNeonI8MMPattern requires LHS/RHS rank <= 2.
    transform.structured.vectorize %tiled_mmt4d vector_sizes [1, 1, 1, 4, 4, 8]
      {create_named_contraction} : !transform.any_op

    transform.apply_patterns to %mmt4d_func {
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.transfer_permutation_patterns
    } : !transform.op<"func.func">

    %mmt4d_func_h = transform.structured.hoist_redundant_vector_transfers %mmt4d_func
      : (!transform.op<"func.func">) -> !transform.op<"func.func">
    %all_loops = transform.structured.match interface{LoopLikeInterface} in %mmt4d_func_h
      : (!transform.op<"func.func">) -> !transform.any_op
    transform.apply_licm to %all_loops : !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %all_loops : !transform.any_op

    transform.apply_patterns to %mmt4d_func_h {
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
      transform.apply_patterns.canonicalization
    } : !transform.op<"func.func">

    %pack = transform.structured.match ops{["linalg.pack"]} in %module : (!transform.any_op) -> !transform.any_op
    %unpack = transform.structured.match ops{["linalg.unpack"]} in %module : (!transform.any_op) -> !transform.any_op

    %tiled_pack_op_p, %loops_pack:2 = transform.structured.tile_using_for %pack tile_sizes [1, 1]
       : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %tiled_unpack_op_p, %loops_unpack:2 = transform.structured.tile_using_for %unpack tile_sizes [4, 4]
       : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %func_op_pack = transform.get_parent_op %tiled_pack_op_p {isolated_from_above} : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op_pack {
      transform.apply_patterns.linalg.decompose_pack_unpack
      transform.apply_patterns.linalg.decompose_pad
    } : !transform.op<"func.func">
    transform.apply_patterns to %func_op_pack {
      transform.apply_patterns.tensor.fold_tensor_subset_ops
      transform.apply_patterns.canonicalization
    } : !transform.op<"func.func">

    %func_op_unpack = transform.get_parent_op %tiled_unpack_op_p {isolated_from_above} : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op_unpack {
      transform.apply_patterns.linalg.decompose_pack_unpack
    } : !transform.op<"func.func">
    transform.apply_patterns to %func_op_unpack {
      transform.apply_patterns.tensor.fold_tensor_subset_ops
      transform.apply_patterns.canonicalization
    } : !transform.op<"func.func">

    %bufferize = transform.bufferization.one_shot_bufferize %module
      {bufferize_function_boundaries=true} : (!transform.any_op) -> !transform.any_op

    %contract = transform.collect_matching @match_contract in %bufferize : (!transform.any_op) -> (!transform.any_op)
    %contract_func = transform.get_parent_op %contract {isolated_from_above} : (!transform.any_op) -> !transform.op<"func.func">

    transform.apply_patterns to %contract_func {
      transform.apply_patterns.tensor.fold_tensor_subset_ops
      transform.apply_patterns.vector.drop_inner_most_unit_dims_from_xfer_ops
      transform.apply_patterns.canonicalization
    } : !transform.op<"func.func">

    // Target FEAT_I8MM directly -- by this point the data is packed and
    // statically shaped, so no masking survives to block the pattern.
    transform.apply_patterns to %contract_func {
      transform.apply_patterns.arm_neon.vector_contract_to_i8mm
    } : !transform.op<"func.func">

    transform.yield
  }

  transform.named_sequence @match_mmt4d(
      %entry: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %entry ["linalg.mmt4d"] : !transform.any_op
    transform.yield %entry : !transform.any_op
  }

  transform.named_sequence @match_contract(
      %entry: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %entry ["vector.contract"] : !transform.any_op
    transform.yield %entry : !transform.any_op
  }
}

func.func private @printMemrefI32(%ptr : tensor<*xi32>)
