// DEFINE: %{td_entry_point} =

// DEFINE: %{compile} = mlir-opt %s \
// DEFINE:    -transform-preload-library='transform-library-paths=%p/td/pack-unpack.mlir' \
// DEFINE:    -transform-interpreter=entry-point=%{td_entry_point} \
// DEFINE:    -lower-vector-mask -convert-vector-to-scf="full-unroll target-rank=0" \
// DEFINE:    -arm-sve-legalize-vector-storage -convert-vector-to-llvm="enable-arm-sve"\
// DEFINE:    -test-lower-to-llvm -o %t
// DEFINE: %{entry_point} = main
// DEFINE: %{run} = %mcr_aarch64_cmd %t -e %{entry_point} -entry-point-result=void  --march=aarch64 --mattr="+sve"\
// DEFINE:    -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%native_mlir_arm_runner_utils

/// Run _without_ vectorization
// REDEFINE: %{td_entry_point} = __transform_main_basic
// RUN: rm -f %t && %{compile} && %{run} | FileCheck %s

/// Run _with_ vectorization
// REDEFINE: %{td_entry_point} = __transform_main_vectorized
// RUN: rm -f %t && %{compile} && %{run} | FileCheck %s

//===----------------------------------------------------------------------===//
/// HIGH-LEVEL OVERVIEW
///
/// End-to-end test for linalg.pack + linalg.unpack where one of the inner tile
/// sizes is scalable.
///
/// Two versions of the transform IR are tested:
///   * without vectorization (see @__transform_main_basic in pack-unpack.mlir)
///   * with vectorization (see @__transform_main_vectorized in pack-unpack.mlir)
///
/// With the payload IR fixed, the runtime output is identical. Note - in both
/// cases the tile sizes are scalable.
///
/// TODO: ATM only linalg.unpack is vectorized. Add linalg.pack vectorization.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// @main
//
// Thin wrapper over the main test function to allow changing the runtime
// vector length via @setArmVLBits (calling setArmVLBits() in a function that
// uses SVE vectors is UB).
//===----------------------------------------------------------------------===//
func.func @main() {
  // Set vscale to 2 (vector width = 256). This will have identical effect to:
  //  * qemu-aarch64 -cpu max,sve-max-vq=2 (...)
  // (If your platform supports it, you can play with other values as well)
  %c256 = arith.constant 256 : i32
  func.call @setArmVLBits(%c256) : (i32) -> ()
  func.call @pack_unpack_with_scalable_inner_tile() : () -> ()

  return
}

//===----------------------------------------------------------------------===//
// @pack_unpack_with_scalable_inner_tile
//
// The main test function that initilaises the matrices an calls pack/unpack
// hooks.
//===----------------------------------------------------------------------===//
func.func @pack_unpack_with_scalable_inner_tile() attributes {no_inline} {
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

//===----------------------------------------------------------------------===//
// @pack_main
//
// Takes the unpacked matrix + inner tile size to use and return the packed matrix.
//===----------------------------------------------------------------------===//
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

//===----------------------------------------------------------------------===//
// @unpack_main
//
/// Takes the packed matrix, unpacks it and returns the result.
//===----------------------------------------------------------------------===//
func.func private @unpack_main(%A_pack : tensor<2x?x4x?xi32>, %inner_tile_size: index) -> tensor<7x12xi32> {
  %A_unpack_empty = tensor.empty() : tensor<7x12xi32>

  %A_unpack = linalg.unpack %A_pack
    inner_dims_pos = [0, 1]
    inner_tiles = [4, %inner_tile_size]
    into %A_unpack_empty : tensor<2x?x4x?xi32> -> tensor<7x12xi32>

  return %A_unpack : tensor<7x12xi32>
}

func.func private @printMemrefI32(%ptr : tensor<*xi32>)
func.func private @setArmVLBits(%bits : i32)
