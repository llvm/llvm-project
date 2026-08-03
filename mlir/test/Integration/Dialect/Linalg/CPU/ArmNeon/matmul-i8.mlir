// REQUIRES: arm-emulator

// DEFINE: %{compile} = mlir-opt %s \
// DEFINE:   -transform-interpreter -test-transform-dialect-erase-schedule \
// DEFINE:   -one-shot-bufferize="bufferize-function-boundaries" -buffer-deallocation-pipeline -cse -canonicalize -convert-vector-to-scf \
// DEFINE:   -convert-vector-to-llvm="enable-arm-neon enable-arm-i8mm" -test-lower-to-llvm \
// DEFINE: -o %t

// DEFINE: %{run} = %mcr_aarch64_cmd %t -e main -entry-point-result=void --march=aarch64 --mattr="+neon,+i8mm" \
// DEFINE:    -shared-libs=%native_mlir_runner_utils,%native_mlir_c_runner_utils

// RUN: rm -f %t && %{compile} && FileCheck %s --input-file=%t -check-prefix CHECK-IR && %{run} | FileCheck %s

// Lowers a vanilla `linalg.matmul` down to Arm's FEAT_I8MM `smmla`.
// `LowerContractionToNeonI8MMPattern` expects the RHS transposed (N-major);
// `transform.structured.transpose_matmul <rhs>` gets us there from a plain matmul.

// CHECK-IR-LABEL: llvm.func @main
// CHECK-IR-COUNT-4: arm_neon.intr.smmla
func.func @main() {
  // A: MxK = 4x8.
  %A = arith.constant dense<[
    [-35, -27, -36, -31,  23, -34,  -8, -33],
    [-20,  17, -32, -47,  37,  22,  -7, -21],
    [ -7, -35,  20,  -4,  39,  46, -23,  40],
    [ 40,  27,  37,  43,  38,  -6,  37,  49]
  ]> : tensor<4x8xi8>

  // B: KxN = 8x4.
  %B = arith.constant dense<[
    [-17, -35, -28, -23],
    [-50, -24,  31,  39],
    [ -1,  37,   3,  48],
    [ 48, -32, -44,  26],
    [-13,  33, -15, -23],
    [ 22,  30, -27,  32],
    [ 39, -11,  22, -39],
    [ 33, -17,  35, -38]
  ]> : tensor<8x4xi8>

  // C: MxN = 4x4, non-zero to also exercise the "+ ACC" part of `smmla`.
  %C = arith.constant dense<[
    [-44,  20,  44, -46],
    [ -8,  25, -34,  26],
    [-20, -36,  -3,  39],
    [-48, -31, -25, -21]
  ]> : tensor<4x4xi32>

  %res = linalg.matmul
      ins(%A, %B : tensor<4x8xi8>, tensor<8x4xi8>)
      outs(%C : tensor<4x4xi32>) -> tensor<4x4xi32>

  // Print and verify the output
  // CHECK-LABEL: NEON: START OF TEST OUTPUT
  vector.print str "NEON: START OF TEST OUTPUT\n"

  // CHECK-NEXT: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
  // CHECK-NEXT: [-1999,    1941,     685,   -2879]
  // CHECK-NEXT: [-3705,    2952,     987,    -685]
  // CHECK-NEXT: [2565,     4157,   -1589,    -357]
  // CHECK-NEXT: [2383,    -2252,      32,   -1365]
  %xf = tensor.cast %res : tensor<4x4xi32> to tensor<*xi32>
  call @printMemrefI32(%xf) : (tensor<*xi32>) -> ()

  // CHECK-NEXT: NEON: END OF TEST OUTPUT
  vector.print str "NEON: END OF TEST OUTPUT\n"

  return
}

module attributes {transform.with_named_sequence} {
  // Tile, vectorize, then lower the `vector.contract` straight to FEAT_I8MM ops.
  transform.named_sequence @tile_and_vectorize_matmul(%func
    : !transform.op<"func.func"> {transform.readonly}) {

    %matmul = transform.structured.match ops{["linalg.matmul"]} in %func
      : (!transform.op<"func.func">) -> !transform.any_op

    %transposed_matmul = transform.structured.transpose_matmul %matmul <rhs>
      : (!transform.any_op) -> (!transform.any_op)

    // M=N=4, K=8: FEAT_I8MM's native tile shape.
    %tiled_matmul, %loops:3 = transform.structured.tile_using_for %transposed_matmul
      tile_sizes [4, 4, 8]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    transform.structured.vectorize %tiled_matmul vector_sizes [4, 4, 8]
      {create_named_contraction} : !transform.any_op

    transform.apply_patterns to %func {
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.lower_masked_transfers
      transform.apply_patterns.vector.sink_ops
    } : !transform.op<"func.func">

    // Lower straight to FEAT_I8MM ops instead of the generic outerproduct
    // path, which would never emit `smmla`.
    transform.apply_patterns to %func {
      transform.apply_patterns.arm_neon.vector_contract_to_i8mm
    } : !transform.op<"func.func">

    transform.yield
  }

  // Apply `tile_and_vectorize_matmul` to every function in the module.
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.op<"func.func">

    transform.foreach %funcs : !transform.op<"func.func"> {
      ^bb0(%func : !transform.op<"func.func">):
        transform.include @tile_and_vectorize_matmul failures(propagate)
          (%func) : (!transform.op<"func.func">) -> ()
    }
    transform.yield
  }
}

func.func private @printMemrefI32(%ptr : tensor<*xi32>)
