// DEFINE: %{compile} =  mlir-opt %s \
// DEFINE:    -transform-interpreter -test-transform-dialect-erase-schedule \
// DEFINE:    -one-shot-bufferize="bufferize-function-boundaries" \
// DEFINE:    -buffer-deallocation-pipeline="private-function-dynamic-ownership" \
// DEFINE:    -cse -canonicalize -test-lower-to-llvm
// DEFINE: %{entry_point} = main
// DEFINE: %{run} = mlir-runner -e %{entry_point} -entry-point-result=void \
// DEFINE:    -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils

// RUN: %{compile} | %{run} | FileCheck %s

/// End-to-end test for the keep_layout_operands option of
/// transform.structured.pack. Packing a linalg.matmul with
/// keep_layout_operands = [2] packs A and B but leaves the accumulator C
/// row-major: it is viewed as [M/MR, MR, N/NR, NR] with tensor.expand_shape and
/// the result is collapsed back, so C is neither packed nor unpacked. The
/// result must equal the one of the plain linalg.matmul. Two cases, each with a
/// reference function (untouched linalg.matmul) and a function that the
/// transform script packs, hence four functions in total:
///   * static shapes, M a multiple of the tile: @matmul and @matmul_keep_c;
///   * M dynamic and not a multiple: @matmul_dyn and @matmul_keep_c_split. The
///     matmul is split at M - M mod MR, the first part keeps C in place and the
///     rest stays a linalg.matmul.
/// The packed functions are selected by the discardable attribute on their
/// linalg.matmul.

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

  // Static, linalg.matmul.
  // CHECK: Unranked Memref
  // CHECK-NEXT: [9,   -5,   -5,   -5,   -5,   2,   2,   16]
  // CHECK-NEXT: [1,   5,   4,   3,   -5,   -6,   -7,   -1]
  // CHECK-NEXT: [-1,   9,   3,   6,   -5,   -9,   -6,   -3]
  // CHECK-NEXT: [-3,   -1,   1,   -6,   10,   -2,   0,   -5]
  // CHECK-NEXT: [0,   -6,   -5,   -4,   -5,   10,   11,   -2]
  // CHECK-NEXT: [13,   -1,   -1,   -1,   -1,   -3,   -3,   11]
  %ref6 = call @matmul(%A6, %B, %C6) : (tensor<6x4xi32>, tensor<4x8xi32>, tensor<6x8xi32>) -> tensor<6x8xi32>
  %ref6_u = tensor.cast %ref6 : tensor<6x8xi32> to tensor<*xi32>
  call @printMemrefI32(%ref6_u) : (tensor<*xi32>) -> ()

  // Static, packed with C kept in place.
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

  // Dynamic M = 7, linalg.matmul.
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
  %ref7 = call @matmul_dyn(%A7_d, %B, %C7_d) : (tensor<?x4xi32>, tensor<4x8xi32>, tensor<?x8xi32>) -> tensor<?x8xi32>
  %ref7_u = tensor.cast %ref7 : tensor<?x8xi32> to tensor<*xi32>
  call @printMemrefI32(%ref7_u) : (tensor<*xi32>) -> ()

  // Dynamic M = 7, split at 6, packed part with C kept in place.
  // CHECK: Unranked Memref
  // CHECK-NEXT: [9,   -5,   -5,   -5,   -5,   2,   2,   16]
  // CHECK-NEXT: [1,   5,   4,   3,   -5,   -6,   -7,   -1]
  // CHECK-NEXT: [-1,   9,   3,   6,   -5,   -9,   -6,   -3]
  // CHECK-NEXT: [-3,   -1,   1,   -6,   10,   -2,   0,   -5]
  // CHECK-NEXT: [0,   -6,   -5,   -4,   -5,   10,   11,   -2]
  // CHECK-NEXT: [13,   -1,   -1,   -1,   -1,   -3,   -3,   11]
  // CHECK-NEXT: [-4,   9,   8,   7,   -1,   -2,   -12,   -6]
  %pk7 = call @matmul_keep_c_split(%A7_d, %B, %C7_d) : (tensor<?x4xi32>, tensor<4x8xi32>, tensor<?x8xi32>) -> tensor<?x8xi32>
  %pk7_u = tensor.cast %pk7 : tensor<?x8xi32> to tensor<*xi32>
  call @printMemrefI32(%pk7_u) : (tensor<*xi32>) -> ()

  return
}

// Static case, reference: not transformed.
func.func private @matmul(%A: tensor<6x4xi32>, %B: tensor<4x8xi32>, %C: tensor<6x8xi32>) -> tensor<6x8xi32> {
  %0 = linalg.matmul ins(%A, %B : tensor<6x4xi32>, tensor<4x8xi32>)
                     outs(%C : tensor<6x8xi32>) -> tensor<6x8xi32>
  return %0 : tensor<6x8xi32>
}

// Static case, same matmul, packed with C kept in place.
func.func private @matmul_keep_c(%A: tensor<6x4xi32>, %B: tensor<4x8xi32>, %C: tensor<6x8xi32>) -> tensor<6x8xi32> {
  %0 = linalg.matmul {keep_c} ins(%A, %B : tensor<6x4xi32>, tensor<4x8xi32>)
                     outs(%C : tensor<6x8xi32>) -> tensor<6x8xi32>
  return %0 : tensor<6x8xi32>
}

// Dynamic case, reference: not transformed.
func.func private @matmul_dyn(%A: tensor<?x4xi32>, %B: tensor<4x8xi32>, %C: tensor<?x8xi32>) -> tensor<?x8xi32> {
  %0 = linalg.matmul ins(%A, %B : tensor<?x4xi32>, tensor<4x8xi32>)
                     outs(%C : tensor<?x8xi32>) -> tensor<?x8xi32>
  return %0 : tensor<?x8xi32>
}

// Dynamic case, same matmul, split at %point and the first part packed with C
// kept in place. The affine.apply computes the split point.
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
    // Step 1: Pack the static matmul, keeping C in place
    // Packs A and B with a 3x4 register tile (K is packed by 1) and leaves the
    // accumulator, operand 2, row-major. The matmul becomes (trimmed):
    //
    //  %A_pack = linalg.pack %A ...  : tensor<6x4xi32> -> tensor<2x4x3x1xi32>
    //  %B_pack = linalg.pack %B ...  : tensor<4x8xi32> -> tensor<2x4x4x1xi32>
    //  %C_view = tensor.expand_shape %C [[0, 1], [2, 3]] output_shape [2, 3, 2, 4]
    //      : tensor<6x8xi32> into tensor<2x3x2x4xi32>
    //  %res = linalg.generic ins(%A_pack, %B_pack) outs(%C_view) ...
    //  %C_out = tensor.collapse_shape %res [[0, 1], [2, 3]]
    //      : tensor<2x3x2x4xi32> into tensor<6x8xi32>
    //
    // There is no linalg.pack of C and no linalg.unpack of the result.
    %mm = transform.structured.match ops{["linalg.matmul"]} attributes {keep_c} in %module : (!transform.any_op) -> !transform.any_op
    %packed = transform.structured.pack %mm packed_sizes = [3, 4, 1] keep_layout_operands = [2]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)

    // Step 2: Split the dynamic matmul, then pack its first part
    // M is dynamic and may not be a multiple of the tile, so linalg.matmul is cut
    // at %point = M - M mod 3 (the affine.apply in @matmul_keep_c_split). The
    // first part has a multiple of 3 rows and is packed like in step 1, with K
    // not packed (size 0). The second part, at most 2 rows, stays a
    // linalg.matmul.
    %mm_dyn = transform.structured.match ops{["linalg.matmul"]} attributes {keep_c_split} in %module : (!transform.any_op) -> !transform.any_op
    %point = transform.structured.match ops{["affine.apply"]} in %module : (!transform.any_op) -> !transform.any_op
    %parts = transform.structured.split %mm_dyn after %point {dimension = 0} : !transform.any_op, !transform.any_op
    %main, %rest = transform.split_handle %parts : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %packed_dyn = transform.structured.pack %main packed_sizes = [3, 4, 0] keep_layout_operands = [2]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)

    // Step 3: Lower linalg.pack
    // Only the packs of A and B are left (C is not packed). They become
    // tensor.pad (a no-op when the size is already a multiple),
    // tensor.expand_shape and linalg.transpose, which the pipeline in the RUN
    // line can bufferize and lower.
    %packs = transform.structured.match ops{["linalg.pack"]} in %module : (!transform.any_op) -> !transform.op<"linalg.pack">
    transform.structured.lower_pack %packs : (!transform.op<"linalg.pack">)
        -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
    transform.yield
  }
}
