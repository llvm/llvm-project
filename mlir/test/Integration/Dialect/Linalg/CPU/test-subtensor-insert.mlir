// RUN: mlir-opt %s -linalg-bufferize -std-bufferize \
// RUN: -tensor-constant-bufferize -tensor-bufferize -func-bufferize \
// RUN: -finalizing-bufferize \
// RUN: -convert-linalg-to-loops -convert-scf-to-std -convert-linalg-to-llvm --convert-memref-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

func @main() {
  %const = arith.constant dense<10.0> : tensor<2xf32>
  %insert_val = arith.constant dense<20.0> : tensor<1xf32>
  %inserted = tensor.insert_slice %insert_val into %const[0][1][1] : tensor<1xf32> into tensor<2xf32>

  %unranked = tensor.cast %inserted : tensor<2xf32> to tensor<*xf32>
  call @print_memref_f32(%unranked) : (tensor<*xf32>) -> ()

  //      CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
  // CHECK-SAME: rank = 1 offset = 0 sizes = [2] strides = [1] data =
  // CHECK-NEXT: [20, 10]

  return
}

func private @print_memref_f32(%ptr : tensor<*xf32>)
