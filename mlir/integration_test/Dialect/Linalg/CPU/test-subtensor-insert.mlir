// RUN: mlir-opt %s -linalg-bufferize -std-bufferize \
// RUN: -tensor-constant-bufferize -tensor-bufferize -func-bufferize \
// RUN: -finalizing-bufferize \
// RUN: -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

func @main() {
  %const = constant dense<10.0> : tensor<2xf32>
  %insert_val = constant dense<20.0> : tensor<1xf32>
  %inserted = subtensor_insert %insert_val into %const[0][1][1] : tensor<1xf32> into tensor<2xf32>

  %unranked = tensor.cast %inserted : tensor<2xf32> to tensor<*xf32>
  call @print_memref_f32(%unranked) : (tensor<*xf32>) -> ()

  //      CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
  // CHECK-SAME: rank = 1 offset = 0 sizes = [2] strides = [1] data =
  // CHECK-NEXT: [20, 10]

  return
}

func private @print_memref_f32(%ptr : tensor<*xf32>)
