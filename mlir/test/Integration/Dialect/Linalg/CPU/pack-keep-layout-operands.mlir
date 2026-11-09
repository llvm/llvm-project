// DEFINE: %{compile} =  mlir-opt %s \
// DEFINE:    -transform-interpreter -test-transform-dialect-erase-schedule \
// DEFINE:    -one-shot-bufferize="bufferize-function-boundaries" \
// DEFINE:    -buffer-deallocation-pipeline="private-function-dynamic-ownership" \
// DEFINE:    -cse -canonicalize -test-lower-to-llvm
// DEFINE: %{entry_point} = main
// DEFINE: %{run} = mlir-runner -e %{entry_point} -entry-point-result=void \
// DEFINE:    -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils

// RUN: %{compile} | %{run} | FileCheck %s

/// Pack a matmul with keep_layout_operands = [2]: A and B are packed, C stays
/// row-major. The dynamic case is split first so that the packed part has a
/// multiple of the tile as M.

func.func @main() {
  %B = arith.constant dense<[
    [ -3,  -2,  -1,   0,   1,   2,   3,  -3],
    [  0,   1,   2,   3,  -3,  -2,  -1,   0],
    [  3,  -3,  -2,  -1,   0,   1,   2,   3],
    [ -1,   0,   1,   2,   3,  -3,  -2,  -1]
  ]> : tensor<4x8xi32>
  %A6 = arith.constant dense<[
    [ -2,   0,   2,  -1],
    [ -1,   1,  -2,   0],
    [  0,   2,  -1,   1],
    [  1,  -2,   0,   2],
    [  2,  -1,   1,  -2],
    [ -2,   0,   2,  -1]
  ]> : tensor<6x4xi32>
  %C6 = arith.constant dense<[
    [ -4,  -3,  -2,  -1,   0,   1,   2,   3],
    [  4,  -4,  -3,  -2,  -1,   0,   1,   2],
    [  3,   4,  -4,  -3,  -2,  -1,   0,   1],
    [  2,   3,   4,  -4,  -3,  -2,  -1,   0],
    [  1,   2,   3,   4,  -4,  -3,  -2,  -1],
    [  0,   1,   2,   3,   4,  -4,  -3,  -2]
  ]> : tensor<6x8xi32>
  %A7 = arith.constant dense<[
    [ -2,   0,   2,  -1],
    [ -1,   1,  -2,   0],
    [  0,   2,  -1,   1],
    [  1,  -2,   0,   2],
    [  2,  -1,   1,  -2],
    [ -2,   0,   2,  -1],
    [ -1,   1,  -2,   0]
  ]> : tensor<7x4xi32>
  %C7 = arith.constant dense<[
    [ -4,  -3,  -2,  -1,   0,   1,   2,   3],
    [  4,  -4,  -3,  -2,  -1,   0,   1,   2],
    [  3,   4,  -4,  -3,  -2,  -1,   0,   1],
    [  2,   3,   4,  -4,  -3,  -2,  -1,   0],
    [  1,   2,   3,   4,  -4,  -3,  -2,  -1],
    [  0,   1,   2,   3,   4,  -4,  -3,  -2],
    [ -1,   0,   1,   2,   3,   4,  -4,  -3]
  ]> : tensor<7x8xi32>

  // Static.
  // CHECK: Unranked Memref
  // CHECK-NEXT: [9,   -5,   -5,   -5,   -5,   2,   2,   16]
  // CHECK-NEXT: [1,   5,   4,   3,   -5,   -6,   -7,   -1]
  // CHECK-NEXT: [-1,   9,   3,   6,   -5,   -9,   -6,   -3]
  // CHECK-NEXT: [-3,   -1,   1,   -6,   10,   -2,   0,   -5]
  // CHECK-NEXT: [0,   -6,   -5,   -4,   -5,   10,   11,   -2]
  // CHECK-NEXT: [13,   -1,   -1,   -1,   -1,   -3,   -3,   11]
  %pk6 = call @matmul_keep_c(%A6, %B, %C6) : (tensor<6x4xi32>, tensor<4x8xi32>, tensor<6x8xi32>) -> tensor<6x8xi32>
  %pk6_u = tensor.cast %pk6 : tensor<6x8xi32> to tensor<*xi32>
  call @printMemrefI32(%pk6_u) : (tensor<*xi32>) -> ()

  // Dynamic M = 7, split at 6.
  // CHECK: Unranked Memref
  // CHECK-NEXT: [9,   -5,   -5,   -5,   -5,   2,   2,   16]
  // CHECK-NEXT: [1,   5,   4,   3,   -5,   -6,   -7,   -1]
  // CHECK-NEXT: [-1,   9,   3,   6,   -5,   -9,   -6,   -3]
  // CHECK-NEXT: [-3,   -1,   1,   -6,   10,   -2,   0,   -5]
  // CHECK-NEXT: [0,   -6,   -5,   -4,   -5,   10,   11,   -2]
  // CHECK-NEXT: [13,   -1,   -1,   -1,   -1,   -3,   -3,   11]
  // CHECK-NEXT: [-4,   9,   8,   7,   -1,   -2,   -12,   -6]
  %A7_d = tensor.cast %A7 : tensor<7x4xi32> to tensor<?x4xi32>
  %C7_d = tensor.cast %C7 : tensor<7x8xi32> to tensor<?x8xi32>
  %pk7 = call @matmul_keep_c_split(%A7_d, %B, %C7_d) : (tensor<?x4xi32>, tensor<4x8xi32>, tensor<?x8xi32>) -> tensor<?x8xi32>
  %pk7_u = tensor.cast %pk7 : tensor<?x8xi32> to tensor<*xi32>
  call @printMemrefI32(%pk7_u) : (tensor<*xi32>) -> ()

  return
}

func.func private @matmul_keep_c(%A: tensor<6x4xi32>, %B: tensor<4x8xi32>, %C: tensor<6x8xi32>) -> tensor<6x8xi32> {
  %0 = linalg.matmul {keep_c} ins(%A, %B : tensor<6x4xi32>, tensor<4x8xi32>)
                     outs(%C : tensor<6x8xi32>) -> tensor<6x8xi32>
  return %0 : tensor<6x8xi32>
}

func.func private @matmul_keep_c_split(%A: tensor<?x4xi32>, %B: tensor<4x8xi32>, %C: tensor<?x8xi32>) -> tensor<?x8xi32> {
  %c0 = arith.constant 0 : index
  %m = tensor.dim %A, %c0 : tensor<?x4xi32>
  %point = affine.apply affine_map<()[s0] -> (s0 - s0 mod 3)>()[%m]
  %0 = linalg.matmul {keep_c_split} ins(%A, %B : tensor<?x4xi32>, tensor<4x8xi32>)
                     outs(%C : tensor<?x8xi32>) -> tensor<?x8xi32>
  return %0 : tensor<?x8xi32>
}

func.func private @printMemrefI32(%ptr : tensor<*xi32>)

module @transforms attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} attributes {keep_c} in %module : (!transform.any_op) -> !transform.any_op
    %packed = transform.structured.pack %mm packed_sizes = [3, 4, 1] keep_layout_operands = [2]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)

    %mm_dyn = transform.structured.match ops{["linalg.matmul"]} attributes {keep_c_split} in %module : (!transform.any_op) -> !transform.any_op
    %point = transform.structured.match ops{["affine.apply"]} in %module : (!transform.any_op) -> !transform.any_op
    %parts = transform.structured.split %mm_dyn after %point {dimension = 0} : !transform.any_op, !transform.any_op
    %main, %rest = transform.split_handle %parts : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %packed_dyn = transform.structured.pack %main packed_sizes = [3, 4, 0] keep_layout_operands = [2]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)

    %packs = transform.structured.match ops{["linalg.pack"]} in %module : (!transform.any_op) -> !transform.op<"linalg.pack">
    transform.structured.lower_pack %packs : (!transform.op<"linalg.pack">)
        -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
    transform.yield
  }
}
