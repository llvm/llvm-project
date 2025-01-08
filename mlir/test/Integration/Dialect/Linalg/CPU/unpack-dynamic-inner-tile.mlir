// DEFINE: %{compile} =  mlir-opt %s \
// DEFINE:  -transform-interpreter -test-transform-dialect-erase-schedule |\
// DEFINE: mlir-opt \
// DEFINE:  -test-lower-to-llvm -o %t
// DEFINE: %{entry_point} = main
// DEFINE: %{run} = mlir-cpu-runner %t -e %{entry_point} -entry-point-result=void \
// DEFINE:    -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils

// RUN: rm -f %t && %{compile} && %{run} | FileCheck %s

/// End-to-end test for tensor.unpack where one of the inner tile sizes is
/// dynamic.

func.func @main() {
  // Allocate and initialise the inputs
  %A_alloc = tensor.empty() : tensor<7x3xi32>

  %A = arith.constant dense<[
  [[[1],
   [2],
   [3],
   [4],
   [5],
   [6],
   [7],
   [123]],
  [[8],
   [9],
   [10],
   [11],
   [12],
   [13],
   [14],
   [123]],
  [[15],
   [16],
   [17],
   [18],
   [19],
   [20],
   [21],
   [123]]]
  ]> : tensor<1x3x8x1xi32>

  %A_cast = tensor.cast %A : tensor<1x3x8x1xi32> to tensor<?x3x?x1xi32>
  func.call @unpack(%A_cast) : (tensor<?x3x?x1xi32>) -> ()

  return
}

func.func private @unpack(%A: tensor<?x3x?x1xi32>) {
  %c1 = arith.constant 1 : index
  %pad_val = arith.constant 123 : i32

  // Dynamic tile size
  %tile_size = arith.constant 8 : index
  %A_unpack_empty = tensor.empty() : tensor<7x3xi32>

  %A_unpack = tensor.unpack %A
    inner_dims_pos = [0, 1]
    inner_tiles = [%tile_size, 1]
    into %A_unpack_empty : tensor<?x3x?x1xi32> -> tensor<7x3xi32>
  %A_cast = tensor.cast %A_unpack : tensor<7x3xi32> to tensor<*xi32>

  // Print the results
  // CHECK: Unranked Memref base@ = 0x{{.*}} rank = 2 offset = 0 sizes = [7, 3] strides = [3, 1] data =
  // CHECK-NEXT: [1,   8,   15],
  // CHECK-NEXT:  [2,   9,   16],
  // CHECK-NEXT:  [3,   10,   17],
  // CHECK-NEXT:  [4,   11,   18],
  // CHECK-NEXT:  [5,   12,   19],
  // CHECK-NEXT:  [6,   13,   20],
  // CHECK-NEXT:  [7,   14,   21]
  call @printMemrefI32(%A_cast) : (tensor<*xi32>) -> ()

  return
}

module @transforms attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.consume}) {
    %pack = transform.structured.match ops{["tensor.unpack"]} in %module : (!transform.any_op) -> !transform.any_op

    // 1. Tile so that we can decompose tensor.pack
    // Ops (see step 2)
    %c8 = transform.param.constant 8 : i64 -> !transform.param<i64>
    %tiled_pack_op_p, %loops:2 = transform.structured.tile_using_for %pack tile_sizes [%c8, 1]
       : (!transform.any_op, !transform.param<i64>) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // 2. Decompose the tiled unpack Op into tensor.extract_slice + tensor.insert_slice:
    %func_op = transform.get_parent_op %tiled_pack_op_p {isolated_from_above} : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.linalg.decompose_pack_unpack
      transform.apply_patterns.linalg.decompose_pad
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
