// DEFINE: %{compile} =  mlir-opt %s \
// DEFINE:  -transform-interpreter -test-transform-dialect-erase-schedule \
// DEFINE:    --lower-vector-mask |\
// DEFINE: mlir-opt -arm-sve-legalize-vector-storage -convert-vector-to-llvm="enable-arm-sve"\
// DEFINE:  -test-lower-to-llvm -o %t
// DEFINE: %{entry_point} = main
// DEFINE: %{run} = %mcr_aarch64_cmd %t -e %{entry_point} -entry-point-result=void  --march=aarch64 --mattr="+sve"\
// DEFINE:    -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%native_mlir_arm_runner_utils

// RUN: rm -f %t && %{compile} && %{run} | FileCheck %s

/// End-to-end test for linalg.pack + linalg.unpack where one of the inner tile sizes is
/// scalable.
/// NOTE: Vectorization has not been enabled yet!


/// The main entry point
func.func @main() {
  // Set vscale to 2 (vector width = 256). This will have identical effect to:
  //  * qemu-aarch64 -cpu max,sve-max-vq=2 (...)
  // (If your platform supports it, you can play with other values as well)
  %c256 = arith.constant 256 : i32
  func.call @setArmVLBits(%c256) : (i32) -> ()
  func.call @test_pack_unpack_scalable_inner_tile() : () -> ()

  return
}

func.func @test_pack_unpack_scalable_inner_tile() attributes {no_inline} {
  // Dynamic/scalable tile size (vscale x 4)
  %c4 = arith.constant 4 : index
  %vs = vector.vscale
  %tile_size = arith.muli %c4, %vs : index

  vector.print str "\nINNER TILE SIZE (run-time value): "
  vector.print %tile_size : index

  // Input matrix. The values and dimension have been selected so that this
  // matrix can be viewed as:
  //  +--------+--------+--------+
  //  |        |        |        |
  //  |  4x4   |  4x4   |  4x4   |
  //  |        |        |        |
  //  +--------+--------+--------+
  //  |        |        |        |
  //  |  3x4   |  3x4   |  3x4   |
  //  |        |        |        |
  //  +--------+--------+--------+
  // This way, after packing, there will be "incomplete"  tiles that will
  // contain the padding value. After unpacking, the padding value should be
  // gone.
  %A_before = arith.constant dense<[
    [1,   1,   1,   1,   2,   2,   2,   2,  3,   3,   3,   3],
    [1,   1,   1,   1,   2,   2,   2,   2,  3,   3,   3,   3],
    [1,   1,   1,   1,   2,   2,   2,   2,  3,   3,   3,   3],
    [1,   1,   1,   1,   2,   2,   2,   2,  3,   3,   3,   3],
    [4,   4,   4,   4,   5,   5,   5,   5,  6,   6,   6,   6],
    [4,   4,   4,   4,   5,   5,   5,   5,  6,   6,   6,   6],
    [4,   4,   4,   4,   5,   5,   5,   5,  6,   6,   6,   6]
  ]> : tensor<7x12xi32>

  // STEP 1: PACK + UNPACK
  // TODO: We should change the order to: Pack+print, Unpack+print. However, that causes the
  // bufferization to fail with:
  //  * 'tensor.cast' op not bufferizable under the given constraints: cannot avoid RaW conflict
  // Investigate and either fix or remove this comment (if impossible to work-around).
  %A_pack = func.call @pack_main(%A_before, %tile_size) : (tensor<7x12xi32>, index) -> tensor<2x?x4x?xi32>
  %A_unpack = func.call @unpack_main(%A_pack, %tile_size) : (tensor<2x?x4x?xi32>, index) -> tensor<7x12xi32>

  // STEP 2: Print the matrices
  vector.print str "\nINPUT MATRIX (before packing)\n"
  %A_before_cast = tensor.cast %A_before : tensor<7x12xi32> to tensor<*xi32>
  call @printMemrefI32(%A_before_cast) : (tensor<*xi32>) -> ()

  vector.print str "\nINPUT MATRIX (after packing)\n"
  %A_pack_cast = tensor.cast %A_pack : tensor<2x?x4x?xi32> to tensor<*xi32>
  // There ought to be at least one pad value inserted into a tile
  // CHECK-LABEL:  (after packing)
  // CHECK:  123
  call @printMemrefI32(%A_pack_cast) : (tensor<*xi32>) -> ()

  vector.print str "\nINPUT MATRIX (after unpacking)\n"
  %A_unpack_cast = tensor.cast %A_unpack : tensor<7x12xi32> to tensor<*xi32>
  // This ought to match the input matrix
  // CHECK-LABEL:  (after unpacking)
  // CHECK:  [1,   1,   1,   1,   2,   2,   2,   2,  3,   3,   3,   3],
  // CHECK:  [1,   1,   1,   1,   2,   2,   2,   2,  3,   3,   3,   3],
  // CHECK:  [1,   1,   1,   1,   2,   2,   2,   2,  3,   3,   3,   3],
  // CHECK:  [1,   1,   1,   1,   2,   2,   2,   2,  3,   3,   3,   3],
  // CHECK:  [4,   4,   4,   4,   5,   5,   5,   5,  6,   6,   6,   6],
  // CHECK:  [4,   4,   4,   4,   5,   5,   5,   5,  6,   6,   6,   6],
  // CHECK:  [4,   4,   4,   4,   5,   5,   5,   5,  6,   6,   6,   6]
  call @printMemrefI32(%A_unpack_cast) : (tensor<*xi32>) -> ()

  return
}

/// Takes the unpacked matrix + inner tile size to use and return the packed matrix.
func.func private @pack_main(%A: tensor<7x12xi32>, %inner_tile_size: index) -> (tensor<2x?x4x?xi32>) {
  // Get the size of dim (we could skip tensor.dim, but this way we can keep it generic)
  %c1 = arith.constant 1 : index
  %dim_1 = tensor.dim %A, %c1 : tensor<7x12xi32>

  // Compute the outer-tile size corresponding to the dynamic inner tile size.
  // NOTE: This step is importantant. While as a user we would only tweak the
  // inner tile sizes, we need to make sure that the outer sizes are updated
  // accordingly.
  %outer_tile_size = arith.ceildivui %dim_1, %inner_tile_size : index

  // NOTE: This is deliberately much larger than the input values in %A_before
  // so that it's easy to spot it in the output.
  %pad_val = arith.constant 123 : i32

  %A_pack_empty = tensor.empty(%outer_tile_size, %inner_tile_size) : tensor<2x?x4x?xi32>

  %A_pack = linalg.pack %A
    padding_value(%pad_val : i32)
    inner_dims_pos = [0, 1]
    inner_tiles = [4, %inner_tile_size]
    into %A_pack_empty : tensor<7x12xi32> -> tensor<2x?x4x?xi32>

  return %A_pack : tensor<2x?x4x?xi32>
}

/// Takes the packed matrix, unpacks it and returns the result.
func.func private @unpack_main(%A_pack : tensor<2x?x4x?xi32>, %inner_tile_size: index) -> tensor<7x12xi32> {
  %A_unpack_empty = tensor.empty() : tensor<7x12xi32>

  %A_unpack = linalg.unpack %A_pack
    inner_dims_pos = [0, 1]
    inner_tiles = [4, %inner_tile_size]
    into %A_unpack_empty : tensor<2x?x4x?xi32> -> tensor<7x12xi32>

  return %A_unpack : tensor<7x12xi32>
}

module @transforms attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.consume}) {
    %pack = transform.structured.match ops{["linalg.pack"]} in %module : (!transform.any_op) -> !transform.any_op
    %unpack = transform.structured.match ops{["linalg.unpack"]} in %module : (!transform.any_op) -> !transform.any_op

    // 1.1 Tile the linalg.pack Op so that we can decompose it into e.g. tensor.pad
    //    and other lower-level Ops (see step 2.1)
    %tiled_pack_op_p, %loops_pack:2 = transform.structured.tile_using_for %pack tile_sizes [1, 1]
       : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // 1.2 Tile the linalg.unpack Op so that we can decompose it into e.g. tensor.pad
    //    and other lower-level Ops (see step 2)
    %tiled_unpack_op_p, %loops_unpack:2 = transform.structured.tile_using_for %unpack tile_sizes [4, 1]
       : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // 2.1. Decompose tiled PackOp into lower-level Ops
    %func_op_pack = transform.get_parent_op %tiled_pack_op_p {isolated_from_above} : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op_pack {
      transform.apply_patterns.linalg.decompose_pack_unpack
      transform.apply_patterns.linalg.decompose_pad
    } : !transform.op<"func.func">

    transform.apply_patterns to %func_op_pack {
      transform.apply_patterns.tensor.fold_tensor_subset_ops
      transform.apply_patterns.canonicalization
    } : !transform.op<"func.func">

    // 2.1. Decompose tiled UnpackOp into lower-level Ops
    %func_op_unpack = transform.get_parent_op %tiled_unpack_op_p {isolated_from_above} : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op_unpack {
      transform.apply_patterns.linalg.decompose_pack_unpack
    } : !transform.op<"func.func">

    transform.apply_patterns to %func_op_unpack {
      transform.apply_patterns.tensor.fold_tensor_subset_ops
      transform.apply_patterns.canonicalization
    } : !transform.op<"func.func">

   // 3. Bufferize before lowering to LLVM
   %bufferize = transform.bufferization.one_shot_bufferize %module
     {bufferize_function_boundaries=true} : (!transform.any_op) -> !transform.any_op

   // 4. Canonicalize
   %func_op_bufferized = transform.structured.match ops{["func.func"]} in %bufferize : (!transform.any_op) -> !transform.op<"func.func">
   transform.apply_patterns to %func_op_bufferized {
     transform.apply_patterns.canonicalization
   } : !transform.op<"func.func">

    transform.yield
  }
}

func.func private @printMemrefI32(%ptr : tensor<*xi32>)
func.func private @setArmVLBits(%bits : i32)
