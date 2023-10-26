// RUN: mlir-opt %s -test-transform-dialect-interpreter -test-transform-dialect-erase-schedule \
// RUN:   -one-shot-bufferize -func-bufferize -cse -canonicalize -convert-vector-to-scf -arm-sve-legalize-vector-storage \
// RUN:   -convert-vector-to-llvm="enable-arm-sve" -test-lower-to-llvm | \
// RUN: %mcr_aarch64_cmd -e=entry -entry-point-result=void --march=aarch64 --mattr="+sve" -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils | \
// RUN: FileCheck %s

func.func @entry() {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %step = arith.constant 1 : index
  %c0_f32 = arith.constant 0.0 : f32

  %vscale = vector.vscale
  %vl_fp = arith.muli %c4, %vscale : index
  %A_alloc = bufferization.alloc_tensor(%c2, %c1) : tensor<?x?xf32>
  %B_alloc = bufferization.alloc_tensor(%c1, %vl_fp) : tensor<?x?xf32>
  %C_alloc = bufferization.alloc_tensor(%c2, %vl_fp) : tensor<?x?xf32>

  %pi = arith.constant  3.14 : f32
  %A = linalg.fill ins(%pi : f32) outs(%A_alloc : tensor<?x?xf32>) -> tensor<?x?xf32>
  %B = linalg.fill ins(%pi : f32) outs(%B_alloc : tensor<?x?xf32>) -> tensor<?x?xf32>
  %C_in = linalg.fill ins(%c0_f32 : f32) outs(%C_alloc : tensor<?x?xf32>) -> tensor<?x?xf32>

  %C_out = linalg.matmul ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>) outs(%C_in: tensor<?x?xf32>) -> tensor<?x?xf32>

  // CHECK-LABEL: SVE: START OF TEST OUTPUT
  vector.print str "SVE: START OF TEST OUTPUT"

  // There are at least 4 x f32 elements in every SVE vector, i.e. 
  //    * %vscale >= 1. 
  // Hence, when checking the outupt there will always be at least 4 elements
  // in every row. For implementations with wider vectors, you should see more
  // elements being printed.
  // CHECK-NEXT: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [2, 16] strides = [16, 1] data =
  // CHECK-NEXT: [9.8596,   9.8596,   9.8596,   9.8596
  // CHECK-NEXT: [9.8596,   9.8596,   9.8596,   9.8596

  %xf = tensor.cast %C_out : tensor<?x?xf32> to tensor<*xf32>
  call @printMemrefF32(%xf) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: SVE: END OF TEST OUTPUT
  vector.print str "SVE: END OF TEST OUTPUT"

  return
}

transform.sequence failures(propagate) {
^bb1(%module_op: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %module_op : (!transform.any_op) -> !transform.any_op
  %func_op = get_parent_op %0 : (!transform.any_op) -> !transform.op<"func.func">
  // The tile sizes match the output matrix sizes
  %1, %loops:3 = transform.structured.tile_using_for %0 [2, [4], 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
  %2 = transform.structured.match ops{["linalg.matmul"]} in %module_op : (!transform.any_op) -> !transform.any_op
  // The vector sizes match the output matrix sizes
  // TOOD: Use variables to re-use "shared" sizes
  transform.structured.vectorize %2 vector_sizes [2, [4], 1] : !transform.any_op

  transform.apply_patterns to %func_op {
    transform.apply_patterns.vector.reduction_to_contract
    transform.apply_patterns.vector.transfer_permutation_patterns
    transform.apply_patterns.vector.lower_masked_transfers
  } : !transform.op<"func.func">
  transform.apply_patterns to %func_op {
    transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
    transform.apply_patterns.vector.lower_outerproduct
  } : !transform.op<"func.func">
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
