// DEFINE: %{td_entry_point} =

// DEFINE: %{compile} =  mlir-opt %s \
// DEFINE:    -transform-interpreter=entry-point=%{td_entry_point} \
// DEFINE:    -test-transform-dialect-erase-schedule \
// DEFINE:    -one-shot-bufferize="bufferize-function-boundaries" \
// DEFINE:    -buffer-deallocation-pipeline="private-function-dynamic-ownership" \
// DEFINE:    -cse -canonicalize -test-lower-to-llvm
// DEFINE: %{entry_point} = main
// DEFINE: %{run} = mlir-runner -e %{entry_point} -entry-point-result=void \
// DEFINE:    -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils

/// Pack all operands (reference)
// REDEFINE: %{td_entry_point} = __transform_main_pack_all
// RUN: %{compile} | %{run} | FileCheck %s

/// Pack A and B only, C keeps its layout
// REDEFINE: %{td_entry_point} = __transform_main_skip_acc
// RUN: %{compile} | %{run} | FileCheck %s

/// Pack B and C only, A keeps its layout
// REDEFINE: %{td_entry_point} = __transform_main_skip_lhs
// RUN: %{compile} | %{run} | FileCheck %s

//===----------------------------------------------------------------------===//
/// HIGH-LEVEL OVERVIEW
///
/// End-to-end test for transform.structured.pack with operands_to_pack. The
/// operands that are not packed keep their layout (they are only reshaped with
/// tensor.expand_shape), so no data moves for them.
///
/// The payload IR is fixed and every transform sequence must produce the same
/// result. M is dynamic (6) and is packed by 3. Since an operand that is not
/// packed cannot be padded, M must be a multiple of 3, which is asserted with
/// assume_dynamic_dims_divisible.
//===----------------------------------------------------------------------===//

func.func @main() {
  %A = arith.constant dense<[
    [ -2,   0,   2,  -1],
    [ -1,   1,  -2,   0],
    [  0,   2,  -1,   1],
    [  1,  -2,   0,   2],
    [  2,  -1,   1,  -2],
    [ -2,   0,   2,  -1]
  ]> : tensor<6x4xi32>
  %B = arith.constant dense<[
    [ -3,  -2,  -1,   0,   1,   2,   3,  -3],
    [  0,   1,   2,   3,  -3,  -2,  -1,   0],
    [  3,  -3,  -2,  -1,   0,   1,   2,   3],
    [ -1,   0,   1,   2,   3,  -3,  -2,  -1]
  ]> : tensor<4x8xi32>
  %C = arith.constant dense<[
    [ -4,  -3,  -2,  -1,   0,   1,   2,   3],
    [  4,  -4,  -3,  -2,  -1,   0,   1,   2],
    [  3,   4,  -4,  -3,  -2,  -1,   0,   1],
    [  2,   3,   4,  -4,  -3,  -2,  -1,   0],
    [  1,   2,   3,   4,  -4,  -3,  -2,  -1],
    [  0,   1,   2,   3,   4,  -4,  -3,  -2]
  ]> : tensor<6x8xi32>

  // CHECK: Unranked Memref
  // CHECK-NEXT: [9,   -5,   -5,   -5,   -5,   2,   2,   16]
  // CHECK-NEXT: [1,   5,   4,   3,   -5,   -6,   -7,   -1]
  // CHECK-NEXT: [-1,   9,   3,   6,   -5,   -9,   -6,   -3]
  // CHECK-NEXT: [-3,   -1,   1,   -6,   10,   -2,   0,   -5]
  // CHECK-NEXT: [0,   -6,   -5,   -4,   -5,   10,   11,   -2]
  // CHECK-NEXT: [13,   -1,   -1,   -1,   -1,   -3,   -3,   11]
  %A_dyn = tensor.cast %A : tensor<6x4xi32> to tensor<?x4xi32>
  %C_dyn = tensor.cast %C : tensor<6x8xi32> to tensor<?x8xi32>
  %res = call @matmul(%A_dyn, %B, %C_dyn) : (tensor<?x4xi32>, tensor<4x8xi32>, tensor<?x8xi32>) -> tensor<?x8xi32>
  %res_unranked = tensor.cast %res : tensor<?x8xi32> to tensor<*xi32>
  call @printMemrefI32(%res_unranked) : (tensor<*xi32>) -> ()

  return
}

func.func private @matmul(%A: tensor<?x4xi32>, %B: tensor<4x8xi32>, %C: tensor<?x8xi32>) -> tensor<?x8xi32> {
  %0 = linalg.matmul ins(%A, %B : tensor<?x4xi32>, tensor<4x8xi32>)
                     outs(%C : tensor<?x8xi32>) -> tensor<?x8xi32>
  return %0 : tensor<?x8xi32>
}

func.func private @printMemrefI32(%ptr : tensor<*xi32>)

module @transforms attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main_pack_all(%module: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %module : (!transform.any_op) -> !transform.any_op
    %packed = transform.structured.pack %mm packed_sizes = [3, 4, 1]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)
    transform.include @lower_pack_unpack failures(propagate) (%module) : (!transform.any_op) -> ()
    transform.yield
  }

  transform.named_sequence @__transform_main_skip_acc(%module: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %module : (!transform.any_op) -> !transform.any_op
    %packed = transform.structured.pack %mm packed_sizes = [3, 4, 1] operands_to_pack = [0, 1]
        assume_dynamic_dims_divisible
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)
    transform.include @lower_pack_unpack failures(propagate) (%module) : (!transform.any_op) -> ()
    transform.yield
  }

  transform.named_sequence @__transform_main_skip_lhs(%module: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %module : (!transform.any_op) -> !transform.any_op
    %packed = transform.structured.pack %mm packed_sizes = [3, 4, 1] operands_to_pack = [1, 2]
        assume_dynamic_dims_divisible
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)
    transform.include @lower_pack_unpack failures(propagate) (%module) : (!transform.any_op) -> ()
    transform.yield
  }

  transform.named_sequence @lower_pack_unpack(%module: !transform.any_op {transform.readonly}) {
    %packs = transform.structured.match ops{["linalg.pack"]} in %module : (!transform.any_op) -> !transform.op<"linalg.pack">
    transform.structured.lower_pack %packs : (!transform.op<"linalg.pack">)
        -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
    %unpacks = transform.structured.match ops{["linalg.unpack"]} in %module : (!transform.any_op) -> !transform.op<"linalg.unpack">
    transform.structured.lower_unpack %unpacks : (!transform.op<"linalg.unpack">)
        -> (!transform.op<"tensor.empty">, !transform.op<"linalg.transpose">,
            !transform.op<"tensor.collapse_shape">, !transform.op<"tensor.extract_slice">,
            !transform.op<"linalg.copy">)
    transform.yield
  }
}
