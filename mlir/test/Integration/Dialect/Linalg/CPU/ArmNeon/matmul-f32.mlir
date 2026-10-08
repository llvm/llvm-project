// REQUIRES: target={{(aarch64|arm64).*}}

// RUN: mlir-opt %s \
// RUN:   -transform-interpreter -test-transform-dialect-erase-schedule \
// RUN:   -one-shot-bufferize="bufferize-function-boundaries" -buffer-deallocation-pipeline -cse -canonicalize -convert-vector-to-scf \
// RUN:   -convert-vector-to-llvm="enable-arm-neon" -test-lower-to-llvm -o %t

// RUN: %mcr_aarch64_cmd %t -e main -entry-point-result=void --march=aarch64 --mattr="+neon" \
// RUN:    -shared-libs=%native_mlir_runner_utils,%native_mlir_c_runner_utils | \
// RUN: FileCheck %s

func.func @main() {
  // Matrix dimensions
  %K = arith.constant 3 : index
  %M = arith.constant 5 : index
  %N = arith.constant 15 : index
  %c0_f32 = arith.constant 0.0 : f32

  // Allocate the matrices
  %A_alloc = bufferization.alloc_tensor(%M, %K) : tensor<?x?xf32>
  %B_alloc = bufferization.alloc_tensor(%K, %N) : tensor<?x?xf32>
  %C_alloc = bufferization.alloc_tensor(%M, %N) : tensor<?x?xf32>

  // Initialise the matrices
  %pi = arith.constant 3.14 : f32
  %A = linalg.fill ins(%pi : f32) outs(%A_alloc : tensor<?x?xf32>) -> tensor<?x?xf32>
  %B = linalg.fill ins(%pi : f32) outs(%B_alloc : tensor<?x?xf32>) -> tensor<?x?xf32>
  %C_in = linalg.fill ins(%c0_f32 : f32) outs(%C_alloc : tensor<?x?xf32>) -> tensor<?x?xf32>

  // Matmul
  %C_out = linalg.matmul ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>) outs(%C_in: tensor<?x?xf32>) -> tensor<?x?xf32>

  // Print and verify the output
  // CHECK-LABEL: NEON: START OF TEST OUTPUT
  vector.print str "NEON: START OF TEST OUTPUT\n"

  // CHECK-NEXT: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [5, 15] strides = [15, 1] data =
  // CHECK-COUNT-5: [29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788]
  %xf = tensor.cast %C_out : tensor<?x?xf32> to tensor<*xf32>
  call @printMemrefF32(%xf) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: NEON: END OF TEST OUTPUT
  vector.print str "NEON: END OF TEST OUTPUT\n"

  return
}

module attributes {transform.with_named_sequence} {
  // Tile and vectorize the matmul.
  transform.named_sequence @tile_and_vectorize_matmul(%func
    : !transform.op<"func.func"> {transform.readonly}) {

    %matmul = transform.structured.match ops{["linalg.matmul"]} in %func
      : (!transform.op<"func.func">) -> !transform.any_op

    // NEON has no scalable vectors: N = 4 matches a full 128-bit register.
    %tiled_matmul, %loops:3 = transform.structured.tile_using_for %matmul tile_sizes [2, 4, 1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    transform.structured.vectorize %tiled_matmul vector_sizes [2, 4, 1] : !transform.any_op

    transform.apply_patterns to %func {
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.lower_masked_transfers
      transform.apply_patterns.vector.sink_ops
    } : !transform.op<"func.func">

    transform.apply_patterns to %func {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
      transform.apply_patterns.vector.lower_outerproduct
    } : !transform.op<"func.func">

    transform.yield
  }

  // Apply `tile_and_vectorize_matmul` to every function in the module.
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.op<"func.func">

    transform.foreach %funcs : !transform.op<"func.func"> {
      ^bb2(%func : !transform.op<"func.func">):
        transform.include @tile_and_vectorize_matmul failures(propagate)
        (%func) : (!transform.op<"func.func">) -> ()
    }
    transform.yield
  }
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
