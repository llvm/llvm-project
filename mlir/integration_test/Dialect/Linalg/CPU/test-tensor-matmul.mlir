// RUN: mlir-opt %s -convert-linalg-on-tensors-to-buffers -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

func @main() {
  %A = constant dense<[[1.0, 2.0], [4.0, 5.0]]> : tensor<2x2xf32>
  %B = constant dense<[[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>
  %C = constant dense<1000.0> : tensor<2x4xf32>

  %D = linalg.matmul ins(%A, %B: tensor<2x2xf32>, tensor<2x4xf32>)
                     init(%C: tensor<2x4xf32>) -> tensor<2x4xf32>

  %unranked = tensor_cast %D : tensor<2x4xf32> to tensor<*xf32>
  call @print_memref_f32(%unranked) : (tensor<*xf32>) -> ()

  //      CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
  // CHECK-SAME: rank = 2 offset = 0 sizes = [2, 4] strides = [4, 1] data =
  // CHECK-NEXT: [1011, 1014, 1017, 1020]
  // CHECK-NEXT: [1029, 1038, 1047, 1056]

  return
}

func @print_memref_f32(%ptr : tensor<*xf32>)
