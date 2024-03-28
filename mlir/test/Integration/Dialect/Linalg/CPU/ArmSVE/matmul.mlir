// DEFINE: %{compile} =  mlir-opt %s \
// DEFINE:    -transform-interpreter -test-transform-dialect-erase-schedule \
// DEFINE:    -one-shot-bufferize="bufferize-function-boundaries" -buffer-deallocation-pipeline -cse -canonicalize -convert-vector-to-scf -arm-sve-legalize-vector-storage \
// DEFINE:    -convert-vector-to-llvm="enable-arm-sve" -test-lower-to-llvm -o %t
// DEFINE: %{entry_point} = matmul_f32
// DEFINE: %{run} = %mcr_aarch64_cmd %t -e %{entry_point} -entry-point-result=void --march=aarch64 --mattr="+sve"\
// DEFINE:    -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils

// RUN: %{compile}

// RUN: %{run} | FileCheck %s --check-prefix=F32

// REDEFINE: %{entry_point} = matmul_mixed_ty
// RUN: %{run} | FileCheck %s --check-prefix=MIXED

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
  // F32-LABEL: SVE: START OF TEST OUTPUT
  vector.print str "SVE: START OF TEST OUTPUT\n"

  // F32-NEXT: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [5, 15] strides = [15, 1] data =
  // F32-COUNT-5: [29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788, 29.5788]
  %xf = tensor.cast %C_out : tensor<?x?xf32> to tensor<*xf32>
  call @printMemrefF32(%xf) : (tensor<*xf32>) -> ()

  // F32-NEXT: SVE: END OF TEST OUTPUT
  vector.print str "SVE: END OF TEST OUTPUT\n"

  return
}

func.func @matmul_mixed_ty() {
  // Matrix dimensions
  %K = arith.constant 3 : index
  %M = arith.constant 5 : index
  %N = arith.constant 15 : index
  %c0_i8 = arith.constant 0 : i8
  %c0_i32 = arith.constant 0 : i32

  // Allocate the matrices
  %A_alloc = bufferization.alloc_tensor(%M, %K) : tensor<?x?xi8>
  %B_alloc = bufferization.alloc_tensor(%K, %N) : tensor<?x?xi8>
  %C_alloc = bufferization.alloc_tensor(%M, %N) : tensor<?x?xi32>

  // Initialise the matrices
  %pi = arith.constant  123 : i8
  %A = linalg.fill ins(%pi : i8) outs(%A_alloc : tensor<?x?xi8>) -> tensor<?x?xi8>
  %B = linalg.fill ins(%pi : i8) outs(%B_alloc : tensor<?x?xi8>) -> tensor<?x?xi8>
  %C_in = linalg.fill ins(%c0_i32 : i32) outs(%C_alloc : tensor<?x?xi32>) -> tensor<?x?xi32>

  // Matmul
  %C_out = linalg.matmul ins(%A, %B: tensor<?x?xi8>, tensor<?x?xi8>) outs(%C_in: tensor<?x?xi32>) -> tensor<?x?xi32>

  // Print and verify the output
  // MIXED-LABEL: SVE: START OF TEST OUTPUT
  vector.print str "SVE: START OF TEST OUTPUT\n"

  // MIXED-NEXT: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [5, 15] strides = [15, 1] data =
  // MIXED-COUNT-5: [45387,   45387,   45387,   45387,   45387,   45387,   45387,   45387,   45387,   45387,   45387,   45387,   45387,   45387,   45387]
  %xf = tensor.cast %C_out : tensor<?x?xi32> to tensor<*xi32>
  call @printMemrefI32(%xf) : (tensor<*xi32>) -> ()

  // MIXED-NEXT: SVE: END OF TEST OUTPUT
  vector.print str "SVE: END OF TEST OUTPUT\n"

  return
}

module attributes {transform.with_named_sequence} {
  // A sequence that will tile and vectorise a Matmul Op
  transform.named_sequence @tile_and_vectorize_matmul(%func
    : !transform.op<"func.func"> {transform.readonly}) {

    // Step 0: Get a handle to the matmul Op
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %func
      : (!transform.op<"func.func">) -> !transform.any_op

    // Step 1: Tile
    %tiled_matmul, %loops:3 = transform.structured.tile_using_for %matmul [2, [4], 1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // Step 2: Vectorize
    transform.structured.vectorize %tiled_matmul vector_sizes [2, [4], 1] : !transform.any_op

    // Step 3: Lower vector.multi_reduction to vector.contract (+ some helpful patterns)
    transform.apply_patterns to %func {
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.lower_masked_transfers
    } : !transform.op<"func.func">

    // Step 4: Lower vector.contract to vector.fma
    transform.apply_patterns to %func {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
      transform.apply_patterns.vector.lower_outerproduct
    } : !transform.op<"func.func">

    transform.yield
  }

  // A sequence that goes over all functions in tis module and applies
  // "tile_and_vectorize_matmul"
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
func.func private @printMemrefI32(%ptr : tensor<*xi32>)
