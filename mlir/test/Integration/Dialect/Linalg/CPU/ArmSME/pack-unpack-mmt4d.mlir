// DEFINE: %{compile} =  mlir-opt %s \
// DEFINE:    -transform-interpreter -test-transform-dialect-erase-schedule \
// DEFINE:    -canonicalize -test-lower-to-arm-sme -convert-vector-to-llvm="enable-arm-sve" \
// DEFINE:    -test-lower-to-llvm
// DEFINE: %{entry_point} = main
// DEFINE: %{run} = %mcr_aarch64_cmd -e %{entry_point} -entry-point-result=void --march=aarch64 --mattr="+sme"\
// DEFINE:    -shared-libs=%native_mlir_runner_utils,%native_mlir_c_runner_utils,%native_arm_sme_abi_shlib

// RUN: %{compile} | %{run} | FileCheck %s

//===----------------------------------------------------------------------===//
/// HIGH-LEVEL OVERVIEW
///
/// End-to-end test for computing matrix-multiplication using linalg.mmt4d. In
/// particular, demonstrates how the following MLIR sequence (implemented in
/// @matmul_via_mmt4d):
///
///   A_pack = linalg.pack A
///   B_pack = linalg.pack B
///   C_pack = linalg.pack C
///   out_pack = linalg.mmt4d(A_pack, B_pack, C_pack)
///   out = linalg.unpack out_pack
///
/// is equivalent to:
///
///  linalg.matmul(A, B, C)
///
/// (implemented in @matmul_via_matmul).
///
/// NOTES ON IMPLEMENTATION
/// 1. The MMT4D example uses _scalable_ tile sizes for data tiling.
///   * The matrix-multiplication dimensions that are scalable: M and N.
///
/// 2. The lowering of linalg.mmt4d leverages scalable vectorisation.
///   * The matrix-multiplication dimensions that are scalable: M, N (to match data
///     tiling configuration).
///
/// 3. Neither `linalg.pack` nor `linalg.unpack` are vectorised ATM.
///
/// 4. The MMT4D and Pack/Unpack Ops are kept in seperate functions to isolate
///    the corresponding lowering and lowering configs.
/// 
/// TODO: Vectorize linalg.pack + linalg.unpack:
///   * https://github.com/llvm/llvm-project/issues/159751
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// @main
//
// The main entry point that computes matrix multiplication via linalg.mmt4d
// and linalg.matmul. Note, the output should be independent of the underlying
// Linalg Op used, as well as SVE vector length.
//===----------------------------------------------------------------------===//
func.func @main() {
  // Allocate and initialise the inputs
  %A_empty = tensor.empty() : tensor<7x16xf32>
  %B_empty = tensor.empty() : tensor<16x13xf32>

  %c3 = arith.constant 3.0 : f32
  %c4 = arith.constant 4.0 : f32
  %A = linalg.fill ins(%c3 : f32) outs(%A_empty : tensor<7x16xf32>) -> tensor<7x16xf32>
  %B = linalg.fill ins(%c4 : f32) outs(%B_empty : tensor<16x13xf32>) -> tensor<16x13xf32>
  %C = arith.constant dense<[
    [ 1.,  8., 15., 22., 29., 36., 43., 50., 57., 64., 71., 78., 85.],
    [ 2.,  9., 16., 23., 30., 37., 44., 51., 58., 65., 72., 79., 86.],
    [ 3., 10., 17., 24., 31., 38., 45., 52., 59., 66., 73., 80., 87.],
    [ 4., 11., 18., 25., 32., 39., 46., 53., 60., 67., 74., 81., 88.],
    [ 5., 12., 19., 26., 33., 40., 47., 54., 61., 68., 75., 82., 89.],
    [ 6., 13., 20., 27., 34., 41., 48., 55., 62., 69., 76., 83., 90.],
    [ 7., 14., 21., 28., 35., 42., 49., 56., 63., 70., 77., 84., 91.]
  ]> : tensor<7x13xf32>

  // VARIANT: Matrix multiplication via linalg.mmt4d
  // CHECK: Unranked Memref
  // CHECK:  [193,   200,   207,   214,   221,   228,   235,   242,   249,   256,   263,   270,   277]
  // CHECK:  [194,   201,   208,   215,   222,   229,   236,   243,   250,   257,   264,   271,   278]
  // CHECK:  [195,   202,   209,   216,   223,   230,   237,   244,   251,   258,   265,   272,   279]
  // CHECK:  [196,   203,   210,   217,   224,   231,   238,   245,   252,   259,   266,   273,   280]
  // CHECK:  [197,   204,   211,   218,   225,   232,   239,   246,   253,   260,   267,   274,   281]
  // CHECK:  [198,   205,   212,   219,   226,   233,   240,   247,   254,   261,   268,   275,   282]
  // CHECK:  [199,   206,   213,   220,   227,   234,   241,   248,   255,   262,   269,   276,   283]
  %C_mmt4d = func.call @matmul_via_mmt4d(%A, %B, %C) : (tensor<7x16xf32>, tensor<16x13xf32>, tensor<7x13xf32>) -> tensor<7x13xf32>
  %C_mmt4d_cast = tensor.cast %C_mmt4d : tensor<7x13xf32> to tensor<*xf32>
  vector.print str "--------------------------\n"
  vector.print str "RESULT FROM linalg.mmt4d:\n"
  vector.print str "--------------------------\n"
  call @printMemrefF32(%C_mmt4d_cast) : (tensor<*xf32>) -> ()

  // VARIANT: Matrix multiplication via linalg.matmul
  // CHECK: Unranked Memref
  // CHECK:  [193,   200,   207,   214,   221,   228,   235,   242,   249,   256,   263,   270,   277]
  // CHECK:  [194,   201,   208,   215,   222,   229,   236,   243,   250,   257,   264,   271,   278]
  // CHECK:  [195,   202,   209,   216,   223,   230,   237,   244,   251,   258,   265,   272,   279]
  // CHECK:  [196,   203,   210,   217,   224,   231,   238,   245,   252,   259,   266,   273,   280]
  // CHECK:  [197,   204,   211,   218,   225,   232,   239,   246,   253,   260,   267,   274,   281]
  // CHECK:  [198,   205,   212,   219,   226,   233,   240,   247,   254,   261,   268,   275,   282]
  // CHECK:  [199,   206,   213,   220,   227,   234,   241,   248,   255,   262,   269,   276,   283]
  %C_matmul = func.call @matmul(%A, %B, %C) : (tensor<7x16xf32>, tensor<16x13xf32>, tensor<7x13xf32>) -> tensor<7x13xf32>
  %C_matmul_cast = tensor.cast %C_matmul : tensor<7x13xf32> to tensor<*xf32>
  vector.print str "\n--------------------------\n"
  vector.print str "RESULT FROM linalg.matmul:\n"
  vector.print str "--------------------------\n"
  call @printMemrefF32(%C_matmul_cast) : (tensor<*xf32>) -> ()

  return
}

//===----------------------------------------------------------------------===//
// @matmul_via_matmul
//
// Implements matrix-multiplication via linalg.matmul
//===----------------------------------------------------------------------===//
func.func private @matmul(%A: tensor<7x16xf32>, %B: tensor<16x13xf32>, %C: tensor<7x13xf32>) -> tensor<7x13xf32> {
  %C_matmul = linalg.matmul ins(%A, %B: tensor<7x16xf32>, tensor<16x13xf32>)
                            outs(%C: tensor<7x13xf32>) -> tensor<7x13xf32>

  return %C_matmul : tensor<7x13xf32>
}

//===----------------------------------------------------------------------===//
// @pack_lhs
//
// Implements packing for the A matrix (LHS) in matrix multiplication. The
// inner tile size for dim M is "scalable": 8 * vscale.
//===----------------------------------------------------------------------===//
func.func private @pack_lhs(%A: tensor<7x16xf32>) -> tensor<1x16x?x1xf32> {
  %pad = arith.constant 0.0 : f32

  %vs = vector.vscale
  %c8 = arith.constant 8 : index
  %vs_c8 = arith.muli %vs, %c8 : index

  %A_pack_empty = tensor.empty(%vs_c8) : tensor<1x16x?x1xf32>
  %A_pack = linalg.pack %A
    padding_value(%pad : f32)
    inner_dims_pos = [0, 1]
    inner_tiles = [%vs_c8, 1]
    into %A_pack_empty : tensor<7x16xf32> -> tensor<1x16x?x1xf32>

  return %A_pack : tensor<1x16x?x1xf32>
}

//===----------------------------------------------------------------------===//
// @pack_rhs
//
// Implements packing for the B matrix (RHS) in matrix multiplication. The
// inner tile size for dim N is "scalable": 8 * vscale.
//===----------------------------------------------------------------------===//
func.func private @pack_rhs(%B: tensor<16x13xf32>) ->  tensor<?x16x?x1xf32> {
  %pad = arith.constant 0.0 : f32

  // Compute the outer tile size.
  %vs = vector.vscale
  %c8 = arith.constant 8 : index
  %vs_c8 = arith.muli %vs, %c8 : index
  %c13 = arith.constant 13 : index
  %outer_tile_size = arith.ceildivui %c13, %vs_c8 : index

  %B_pack_empty = tensor.empty(%outer_tile_size, %vs_c8) : tensor<?x16x?x1xf32>
  %B_pack = linalg.pack %B
     padding_value(%pad : f32)
     outer_dims_perm = [1, 0]
     inner_dims_pos = [1, 0]
     inner_tiles = [%vs_c8, 1]
     into %B_pack_empty : tensor<16x13xf32> -> tensor<?x16x?x1xf32>

  return %B_pack : tensor<?x16x?x1xf32>
}

//===----------------------------------------------------------------------===//
// @pack_acc
//
// Implements packing for the C matrix (accumulator) in matrix multiplication.
// The inner tile sizes are "scalable": 8 * vscale, 8 * vscale
//===----------------------------------------------------------------------===//
func.func private @pack_acc(%C: tensor<7x13xf32>) -> tensor<1x?x?x?xf32> {
  %pad = arith.constant 0.0 : f32

  // Compute the outer tile size.
  %c13 = arith.constant 13 : index
  %vs = vector.vscale
  %c8 = arith.constant 8 : index
  %vs_c8 = arith.muli %vs, %c8 : index
  %outer_tile_size = arith.ceildivui %c13, %vs_c8 : index

  %C_pack_empty = tensor.empty(%outer_tile_size, %vs_c8, %vs_c8) : tensor<1x?x?x?xf32>
  %C_pack = linalg.pack %C
    padding_value(%pad : f32)
    outer_dims_perm = [0, 1]
    inner_dims_pos = [0, 1]
    inner_tiles = [%vs_c8, %vs_c8] into %C_pack_empty : tensor<7x13xf32> -> tensor<1x?x?x?xf32>

  return %C_pack : tensor<1x?x?x?xf32>
}

//===----------------------------------------------------------------------===//
// @unpack_acc
//
// Implements unpacking for the C matrix (accumulator) in matrix
// multiplication. The inner tile sizes are "scalable": 8 * vscale, 8 * vscale
//===----------------------------------------------------------------------===//
func.func private @unpack_acc(%C_packed: tensor<1x?x?x?xf32>) -> tensor<7x13xf32> {
  %vs = vector.vscale
  %c8 = arith.constant 8 : index
  %vs_c8 = arith.muli %vs, %c8 : index

  %C_out_empty = tensor.empty() : tensor<7x13xf32>
  %C_out_unpack = linalg.unpack %C_packed
    outer_dims_perm = [0, 1]
    inner_dims_pos = [0, 1]
    inner_tiles = [%vs_c8, %vs_c8]
    into %C_out_empty : tensor<1x?x?x?xf32> -> tensor<7x13xf32>

  return %C_out_unpack: tensor<7x13xf32>
}

//===----------------------------------------------------------------------===//
// @matmul_via_mmt4d
//
// Implements matrix-multiplication via linalg.mmt4d
//===----------------------------------------------------------------------===//
func.func private @matmul_via_mmt4d(%A: tensor<7x16xf32>, %B: tensor<16x13xf32>, %C: tensor<7x13xf32>) -> tensor<7x13xf32> {
  // Pack input matrices
  %A_pack = func.call @pack_lhs(%A): (tensor<7x16xf32>) -> tensor<1x16x?x1xf32>
  %B_pack = func.call @pack_rhs(%B): (tensor<16x13xf32>) -> tensor<?x16x?x1xf32>
  %C_pack = func.call @pack_acc(%C): (tensor<7x13xf32>) -> tensor<1x?x?x?xf32>

  // MMT4D
  %mmt4d = linalg.mmt4d ins(%A_pack, %B_pack : tensor<1x16x?x1xf32>, tensor<?x16x?x1xf32>) outs(%C_pack : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>

  // Unpack the output
  %C_out_unpack = func.call @unpack_acc(%mmt4d) : (tensor<1x?x?x?xf32>) -> tensor<7x13xf32>

  return %C_out_unpack : tensor<7x13xf32>
}

//===----------------------------------------------------------------------===//
// TD Sequence
//===----------------------------------------------------------------------===//
module @transforms attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.consumed}) {
    //==========================================================================
    // HANDLE MMT4D
    //==========================================================================
    %mmt4d = transform.collect_matching @match_mmt4d in %module : (!transform.any_op) -> (!transform.any_op)

    // Step 1: Tile
    // Tile parallel dims (note, the M, N dims are scalable!)
    %tiled_mmt4d_parallel, %_:4 = transform.structured.tile_using_for %mmt4d tile_sizes [1, 1, 0, [8], [8], 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    // Tile reduction dims
    %tiled_mmt4d, %loop_k = transform.structured.tile_using_for %tiled_mmt4d_parallel tile_sizes [0, 0, 1, 0, 0, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)

    // Step 2: Vectorize linalg.mmt4d (note, the M, N dims are scalable!)
    transform.structured.vectorize %tiled_mmt4d vector_sizes  [1, 1, 1, [8], [8], 1] {assume_dynamic_dims_match_vec_sizes, create_named_contraction}
      : !transform.any_op

    // Step 3: Lower vector.mask %mask { vector.transfer_* } to vector.transfer_* %mask
    transform.apply_patterns to %loop_k {
      transform.apply_patterns.vector.lower_masked_transfers
    } : !transform.op<"scf.for">

    // Step 4: Hoist the C accumulator load/store out of the k-loop while still
    // in tensor form, so transfer_write has a result value the loop can yield.
    transform.apply_licm to %loop_k : !transform.op<"scf.for">
    transform.loop.hoist_loop_invariant_subsets %loop_k : !transform.op<"scf.for">

    // Lower to outerproduct
    %func_pre = transform.structured.match ops{["func.func"]} in %module
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_pre {
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_patterns to %func_pre {
      // TODO: `create_named_contraction` above lowers directly to a named contraction. If we want to remove
      // `transform.apply_patterns.vector.reduction_to_contract` below, `CombineContractBroadcastMask`
      // will have to be moved to a dedicated TD Op.
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
      transform.apply_patterns.vector.lower_contraction
          lowering_strategy = "outerproduct"
      transform.apply_patterns.vector.drop_unit_dims_with_shape_cast
      transform.apply_patterns.canonicalization
    } {apply_cse} : !transform.any_op

   //==========================================================================
   // HANDLE PACK + UNPACK
   //==========================================================================
    %pack = transform.structured.match ops{["linalg.pack"]} in %module : (!transform.any_op) -> !transform.any_op
    %unpack = transform.structured.match ops{["linalg.unpack"]} in %module : (!transform.any_op) -> !transform.any_op

    // 1.1 Tile the linalg.pack Op so that we can decompose it into e.g. tensor.pad
    //    and other lower-level Ops (see step 2.1)
    %tiled_pack_op_p, %loops_pack:2 = transform.structured.tile_using_for %pack tile_sizes [1, 1]
       : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // 1.2 Tile the linalg.unpack Op so that we can decompose it into e.g. tensor.pad
    //    and other lower-level Ops (see step 2). Tile by [8*vscale, 8*vscale]
    //    (scalable, equal to the inner tile sizes) so the tiled outer dims are
    //    statically 1 and DecomposeOuterUnitDimsUnPackOp can fire.
    %tiled_unpack_op_p, %loops_unpack:2 = transform.structured.tile_using_for %unpack tile_sizes [[8], [8]]
        inner_tile_alignments = [Equal, Equal]
       : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // 2.1. Decompose tiled PackOp into lower-level Ops + simplify
    %func_op_pack = transform.get_parent_op %tiled_pack_op_p {isolated_from_above} : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op_pack {
      transform.apply_patterns.linalg.decompose_pack_unpack
      transform.apply_patterns.linalg.decompose_pad
    } : !transform.op<"func.func">

    transform.apply_patterns to %func_op_pack {
      transform.apply_patterns.tensor.fold_tensor_subset_ops
      transform.apply_patterns.canonicalization
    } : !transform.op<"func.func">

    // 2.2. Decompose tiled UnpackOp into lower-level Ops + simplify
    %func_op_unpack = transform.get_parent_op %tiled_unpack_op_p {isolated_from_above} : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op_unpack {
      transform.apply_patterns.linalg.decompose_pack_unpack
    } : !transform.op<"func.func">

    transform.apply_patterns to %func_op_unpack {
      transform.apply_patterns.tensor.fold_tensor_subset_ops
      transform.apply_patterns.canonicalization
    } : !transform.op<"func.func">

   //==========================================================================
   // BUFFERIZATION
   //==========================================================================
   %bufferize = transform.bufferization.one_shot_bufferize layout{IdentityLayoutMap} %module
     {bufferize_function_boundaries=true} : (!transform.any_op) -> !transform.any_op
   %func = transform.structured.match ops{["func.func"]} in %bufferize
     : (!transform.any_op) -> !transform.any_op

   transform.apply_patterns to %func {
     transform.apply_patterns.vector.rank_reducing_subview_patterns
     transform.apply_patterns.vector.drop_unit_dims_with_shape_cast
     transform.apply_patterns.canonicalization
   } {apply_cse} : !transform.any_op

   transform.yield
   }

   //==========================================================================
   // TD MATCHERS (helper hooks)
   //==========================================================================
  transform.named_sequence @match_mmt4d(
      %entry: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %entry ["linalg.mmt4d"] : !transform.any_op
    transform.yield %entry : !transform.any_op
  }
}

//===----------------------------------------------------------------------===//
// Function signatures
//===----------------------------------------------------------------------===//
func.func private @printMemrefF32(%ptr : tensor<*xf32>)
