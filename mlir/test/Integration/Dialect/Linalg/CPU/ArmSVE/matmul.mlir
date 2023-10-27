// RUN: mlir-opt %s -test-transform-dialect-interpreter -test-transform-dialect-erase-schedule \
// RUN:   -one-shot-bufferize -func-bufferize -cse -canonicalize -convert-vector-to-scf -arm-sve-legalize-vector-storage \
// RUN:   -convert-vector-to-llvm="enable-arm-sve" -test-lower-to-llvm | \
// RUN: %mcr_aarch64_cmd -e=matmul_f32 -entry-point-result=void --march=aarch64 --mattr="+sve" -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils | \
// RUN: FileCheck %s

func.func @matmul_f32() {
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
  %pi = arith.constant  3.14 : f32
  %A = linalg.fill ins(%pi : f32) outs(%A_alloc : tensor<?x?xf32>) -> tensor<?x?xf32>
  %B = linalg.fill ins(%pi : f32) outs(%B_alloc : tensor<?x?xf32>) -> tensor<?x?xf32>
  %C_in = linalg.fill ins(%c0_f32 : f32) outs(%C_alloc : tensor<?x?xf32>) -> tensor<?x?xf32>

  // Matmul
  %C_out = linalg.matmul ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>) outs(%C_in: tensor<?x?xf32>) -> tensor<?x?xf32>

  // Print and verify the output
  // CHECK-LABEL: SVE: START OF TEST OUTPUT
  vector.print str "SVE: START OF TEST OUTPUT"

  // CHECK-NEXT: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [5, 15] strides = [15, 1] data =
  // CHECK-COUNT-5: [29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788]
  %xf = tensor.cast %C_out : tensor<?x?xf32> to tensor<*xf32>
  call @printMemrefF32(%xf) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: SVE: END OF TEST OUTPUT
  vector.print str "SVE: END OF TEST OUTPUT"

  return
}

transform.sequence failures(propagate) {
^bb1(%module_op: !transform.any_op):
  // Step 1: Tile
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %module_op : (!transform.any_op) -> !transform.any_op
  %func_op = get_parent_op %matmul : (!transform.any_op) -> !transform.op<"func.func">
  %module_with_tiled_loops, %loops:3 = transform.structured.tile_using_for %matmul [2, [4], 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

  // Step 2: Vectorize
  %tiled_matmul = transform.structured.match ops{["linalg.matmul"]} in %module_with_tiled_loops : (!transform.any_op) -> !transform.any_op
  transform.structured.vectorize %tiled_matmul vector_sizes [2, [4], 1] : !transform.any_op

  // Step 3: Lower vector.multi_reduction to vector.contract (+ some helpful patterns)
  transform.apply_patterns to %func_op {
    transform.apply_patterns.vector.reduction_to_contract
    transform.apply_patterns.vector.transfer_permutation_patterns
    transform.apply_patterns.vector.lower_masked_transfers
  } : !transform.op<"func.func">

  // Step 4: Lower vector.contract to vector.fma
  transform.apply_patterns to %func_op {
    transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
    transform.apply_patterns.vector.lower_outerproduct
  } : !transform.op<"func.func">
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
