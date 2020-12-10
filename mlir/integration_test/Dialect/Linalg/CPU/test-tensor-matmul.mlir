// RUN: mlir-opt %s -linalg-bufferize -std-bufferize -tensor-constant-bufferize \
// RUN: -func-bufferize -finalizing-bufferize  -convert-linalg-to-loops \
// RUN: -convert-linalg-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

// RUN: mlir-opt %s  -linalg-tile="linalg-tile-sizes=1,2,3" -linalg-bufferize \
// RUN: -scf-bufferize -std-bufferize -tensor-constant-bufferize -func-bufferize \
// RUN: -finalizing-bufferize -convert-linalg-to-loops -convert-scf-to-std \
// RUN: -convert-linalg-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

func @main() {
  %A = constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %B = constant dense<[[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [9.0, 10.0, 11.0, 12.0]]> : tensor<3x4xf32>
  %C = constant dense<1000.0> : tensor<2x4xf32>

  %D = linalg.matmul ins(%A, %B: tensor<2x3xf32>, tensor<3x4xf32>)
                     init(%C: tensor<2x4xf32>) -> tensor<2x4xf32>

  %unranked = tensor_cast %D : tensor<2x4xf32> to tensor<*xf32>
  call @print_memref_f32(%unranked) : (tensor<*xf32>) -> ()

  //      CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
  // CHECK-SAME: rank = 2 offset = 0 sizes = [2, 4] strides = [4, 1] data =
  // CHECK-NEXT: [1038,   1044,   1050,   1056]
  // CHECK-NEXT: [1083,   1098,   1113,   1128]

  return
}

func private @print_memref_f32(%ptr : tensor<*xf32>)
