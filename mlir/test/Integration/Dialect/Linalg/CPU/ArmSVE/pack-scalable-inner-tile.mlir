// REQUIRES: arm-emulator

// This test is a clone of pack-dynamic-inner-tile.mlir, but the inner tile is
// vector.vscale * %c8 rather than %c8. In order to demonstrate the impact of
// using scalable vectors, vscale is set to 2 so that that the run-time tile
// size is [16, 1] rather than [8, 1].
//
// Note that you can also tweak the size of vscale by passing this flag to
// QEMU:
//  * -cpu max,sve-max-vq=[1-16]
// (select the value between 1 and 16).

// DEFINE: %{compile} =  mlir-opt %s \
// DEFINE:    --transform-interpreter --test-transform-dialect-erase-schedule \
// DEFINE:    --lower-vector-mask \
// DEFINE:    -canonicalize -cse --convert-vector-to-scf \
// DEFINE:    -arm-sve-legalize-vector-storage -convert-vector-to-llvm="enable-arm-sve" -test-lower-to-llvm -o %t

// DEFINE: %{entry_point} = main
// DEFINE: %{run} = %mcr_aarch64_cmd %t -e %{entry_point} -entry-point-result=void --march=aarch64 --mattr="+sve"\
// DEFINE:    -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%native_mlir_arm_runner_utils

// RUN: rm -f %t && %{compile} &&  %{run} |  FileCheck %s

/// End-to-end test for tensor.pack where one of the inner tile sizes is
/// scalable.

func.func @main() {
  // Allocate and initialise the inputs
  %A_alloc = tensor.empty() : tensor<7x16xi32>

  %A = arith.constant dense<[
    [ 1,  8, 15, 22, 29, 36, 43, 50, 57, 64, 71, 78, 85, 92, 99 , 106],
    [ 2,  9, 16, 23, 30, 37, 44, 51, 58, 65, 72, 79, 86, 93, 100, 107],
    [ 3, 10, 17, 24, 31, 38, 45, 52, 59, 66, 73, 80, 87, 94, 101, 108],
    [ 4, 11, 18, 25, 32, 39, 46, 53, 60, 67, 74, 81, 88, 95, 102, 109],
    [ 5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110],
    [ 6, 13, 20, 27, 34, 41, 48, 55, 62, 69, 76, 83, 90, 97, 104, 111],
    [ 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 112]
  ]> : tensor<7x16xi32>

  func.call @pack(%A) : (tensor<7x16xi32>) -> ()

  return
}

func.func private @pack(%A: tensor<7x16xi32>) {
  %c1 = arith.constant 1 : index
  %pad_val = arith.constant 123 : i32

  // Set vscale to 2 (vector width = 256). This will have identical effect to:
  //  * qemu-aarch64 -cpu max,sve-max-vq=2 (...)
  %c256 = arith.constant 256 : i32
  func.call @setArmVLBits(%c256) : (i32) -> ()

  // Scalable tile size
  %vs = vector.vscale
  %c8 = arith.constant 8 : index
  %tile_size = arith.muli %c8, %vs : index

  %A_pack_empty = tensor.empty(%c1, %tile_size) : tensor<?x16x?x1xi32>

  %A_pack = tensor.pack %A
    padding_value(%pad_val : i32)
    inner_dims_pos = [0, 1]
    inner_tiles = [%tile_size, 1]
    into %A_pack_empty : tensor<7x16xi32> -> tensor<?x16x?x1xi32>

  %A_cast = tensor.cast %A_pack : tensor<?x16x?x1xi32> to tensor<*xi32>

  // Print the results
  // CHECK: Unranked Memref base@ = 0{{.*}} rank = 4 offset = 0 sizes = [1, 16, 16, 1] strides = [256, 16, 1, 1] data =
  // Tile 1: ((vscale x 8) x 1)
  // CHECK-NEXT:  1
  // CHECK-NEXT:  2
  // CHECK-NEXT:  3
  // CHECK-NEXT:  4
  // CHECK-NEXT:  5
  // CHECK-NEXT:  6
  // CHECK-NEXT:  7
  // Expect pad value after 7 elements
  // CHECK-NEXT:  123
  // CHECK-NEXT:  123
  // CHECK-NEXT:  123
  // CHECK-NEXT:  123
  // CHECK-NEXT:  123
  // CHECK-NEXT:  123
  // CHECK-NEXT:  123
  // CHECK-NEXT:  123
  // CHECK-NEXT:  123
  // Tile 2: ((vscale x 8) x 1)
  // CHECK-NEXT:  8
  // CHECK-NEXT:  9
  // CHECK-NEXT:  10
  // CHECK-NEXT:  11
  // CHECK-NEXT:  12
  // CHECK-NEXT:  13
  // CHECK-NEXT:  14
  // Expect pad value after further 7 elements
  // CHECK-NEXT:  123
  // CHECK-NEXT:  123
  // CHECK-NEXT:  123
  // CHECK-NEXT:  123
  // CHECK-NEXT:  123
  // CHECK-NEXT:  123
  // CHECK-NEXT:  123
  // CHECK-NEXT:  123
  // CHECK-NEXT:  123
  // Tile 3: ((vscale x 8) x 1)
  // CHECK-NEXT:  15
  // CHECK-NEXT:  16
  // ...
  call @printMemrefI32(%A_cast) : (tensor<*xi32>) -> ()

  return
}

module @transforms attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.consume}) {
    %pack = transform.structured.match ops{["tensor.pack"]} in %module : (!transform.any_op) -> !transform.any_op

    // 1. Tile so that we can decompose tensor.pack into tensor.pad and other
    // Ops (see step 2)
    %tiled_pack_op_p, %loops:2 = transform.structured.tile_using_for %pack tile_sizes [1, 1]
       : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // 2. Decompose the tiled pack Op into (trimmed for brevity):
    //
    //  %padded = tensor.pad %slice_of_A (..) :
    //      tensor<?x?xi32> to tensor<8x1xi32>
    //  %inserted_slice = tensor.insert_slice %padded into %slice_of_A_pack (...) :
    //      tensor<8x1xi32> into tensor<1x1x?x1xi32>
    //
    // (NOTE: no tile is transposed, hence no linalg.transpose)
    //
    // This is followed by this decomposition of the pad Op:
    //
    //  %c123_i32 = arith.constant 123 : i32
    //  %slice_of_A = tensor.extract_slice %A[%3, %arg3] [%4, %5] [1, 1] :
    //    tensor<7x16xi32> to tensor<?x?xi32>
    //  %empty = tensor.empty() : tensor<8x1xi32>
    //  %fill = linalg.fill ins(%c123_i32 : i32) outs(%empty :
    //    tensor<8x1xi32>) -> tensor<8x1xi32>
    //  %inserted_slice = tensor.insert_slice %slice_of_A into %fill[0, 0] [%4, %5] [1, 1] :
    //    tensor<?x?xi32> into tensor<8x1xi32>
    //
    %func_op = transform.get_parent_op %tiled_pack_op_p {isolated_from_above} : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.linalg.decompose_pack_unpack
      transform.apply_patterns.linalg.decompose_pad
    } : !transform.op<"func.func">

    // 3. Vectorize linalg.fill.
    // Vector sizes match the inner tiles in the payload IR.
    %fill = transform.structured.match ops{["linalg.fill"]} in %func_op : (!transform.op<"func.func">) -> !transform.any_op
    transform.structured.vectorize %fill vector_sizes [[8], 1] : !transform.any_op

    transform.apply_patterns to %func_op {
      transform.apply_patterns.tensor.fold_tensor_subset_ops
      transform.apply_patterns.canonicalization
    } : !transform.op<"func.func">

    // 3. Bufferize before lowering to LLVM
    %bufferize = transform.bufferization.one_shot_bufferize %module
      {bufferize_function_boundaries=true} : (!transform.any_op) -> !transform.any_op

    // 4. Canonicalize + rank-reducing patters (to get rid of the trailing unit
    // dim).
    %func_op_bufferized = transform.structured.match ops{["func.func"]} in %bufferize : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op_bufferized {
      transform.apply_patterns.vector.rank_reducing_subview_patterns
      transform.apply_patterns.vector.drop_unit_dims_with_shape_cast
      transform.apply_patterns.canonicalization
    } : !transform.op<"func.func">

    transform.yield
  }
}

func.func private @printMemrefI32(%ptr : tensor<*xi32>)
func.func private @setArmVLBits(%bits : i32)
