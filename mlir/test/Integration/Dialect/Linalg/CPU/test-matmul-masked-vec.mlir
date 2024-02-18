// RUN: mlir-opt %s -transform-interpreter -test-transform-dialect-erase-schedule -one-shot-bufferize -func-bufferize -lower-vector-mask --test-lower-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void --shared-libs=%mlir_c_runner_utils,%mlir_runner_utils | \
// RUN: FileCheck %s

func.func private @printMemrefF32(%ptr : tensor<*xf32>)

func.func @main() {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  %A = arith.constant dense<[
          [ 1.1, 2.1 ],
          [ 1.2, 2.2 ],
          [ 1.3, 2.3 ],
          [ 1.4, 2.4 ],
          [ 1.5, 2.5 ],
          [ 1.6, 2.6 ],
          [ 1.7, 2.7 ],
          [ 1.8, 2.8 ]
      ]> : tensor<8x2xf32>
  %B = arith.constant dense<[
          [ 10.1, 11.1, 12.1, 13.1 ],
          [ 10.2, 11.2, 12.2, 13.2 ]
      ]> : tensor<2x4xf32>
  %C_dyn = bufferization.alloc_tensor(%c8, %c4) : tensor<?x?xf32>

  %A_dyn = tensor.cast %A : tensor<8x2xf32> to tensor<?x?xf32>
  %B_dyn = tensor.cast %B : tensor<2x4xf32> to tensor<?x?xf32>

  %c0_i32 = arith.constant  0 : i32
  %C_init = linalg.fill ins(%c0_i32 : i32) outs(%C_dyn : tensor<?x?xf32>) -> tensor<?x?xf32>

  %res = linalg.matmul ins(%A_dyn, %B_dyn: tensor<?x?xf32>, tensor<?x?xf32>)
            outs(%C_init: tensor<?x?xf32>) -> tensor<?x?xf32>
  %xf = tensor.cast %res : tensor<?x?xf32> to tensor<*xf32>

  // CHECK:      {{\[}}[32.53,   35.73,   38.93,   42.13],
  // CHECK-NEXT: [34.56,   37.96,   41.36,   44.76],
  // CHECK-NEXT: [36.59,   40.19,   43.79,   47.39],
  // CHECK-NEXT: [38.62,   42.42,   46.22,   50.02],
  // CHECK-NEXT: [0,   0,   0,   0],
  // CHECK-NEXT: [0,   0,   0,   0],
  // CHECK-NEXT: [0,   0,   0,   0],
  // CHECK-NEXT: [0,   0,   0,   0]]
  call @printMemrefF32(%xf) : (tensor<*xf32>) -> ()

  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %func_op = transform.get_parent_op %0 : (!transform.any_op) -> !transform.op<"func.func">
    transform.structured.vectorize %0 vector_sizes [4, 4, 2] : !transform.any_op
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerreduction"
    } : !transform.op<"func.func">
    transform.yield
  }
}
