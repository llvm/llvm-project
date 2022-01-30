// RUN: mlir-opt %s -linalg-bufferize -std-bufferize \
// RUN: -arith-bufferize -tensor-bufferize -func-bufferize \
// RUN: -finalizing-bufferize -buffer-deallocation \
// RUN: -convert-linalg-to-loops -convert-scf-to-std -convert-linalg-to-llvm -convert-memref-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s


func @main() {
  %const = arith.constant dense<[[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]]> : tensor<1x2x3xf32>
  %dynamic = tensor.cast %const: tensor<1x2x3xf32> to tensor<1x?x3xf32>
  %offset = arith.constant 2 : index
  %cst = arith.constant 2.3 : f32
  %c0 = arith.constant 0 : index
  %out = tensor.pad %dynamic low[%c0, %offset, %c0] high[%c0, %c0, %offset]  {
  ^bb0(%gen_arg1: index, %gen_arg2: index, %gen_arg3: index):
    tensor.yield %cst : f32
  } : tensor<1x?x3xf32> to tensor<1x?x?xf32>
  %unranked = tensor.cast %out: tensor<1x?x?xf32> to tensor<*xf32>
  call @print_memref_f32(%unranked) : (tensor<*xf32>) -> ()

  //      CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
  // CHECK-SAME: rank = 3 offset = 0 sizes = [1, 4, 5] strides = [20, 5, 1] data =
  // CHECK-NEXT{LITERAL}: [[[2.3,    2.3,    2.3,    2.3,    2.3],
  // CHECK-NEXT: [2.3,    2.3,    2.3,    2.3,    2.3],
  // CHECK-NEXT: [1,    2,    3,    2.3,    2.3],
  // CHECK-NEXT: [2,    3,    4,    2.3,    2.3]]]

  return
}

func private @print_memref_f32(%ptr : tensor<*xf32>)
