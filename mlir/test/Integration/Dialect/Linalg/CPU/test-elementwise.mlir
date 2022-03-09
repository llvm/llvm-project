// RUN: mlir-opt %s -convert-elementwise-to-linalg \
// RUN: -arith-bufferize -linalg-bufferize -tensor-bufferize \
// RUN: -func-bufferize -buffer-deallocation -convert-linalg-to-loops \
// RUN: -convert-linalg-to-llvm --convert-memref-to-llvm -convert-func-to-llvm \
// RUN: -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

func @main() {
  %a = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
  %b = arith.constant dense<[10.0, 20.0, 30.0]> : tensor<3xf32>

  %addf = arith.addf %a, %b : tensor<3xf32>
  %addf_unranked = tensor.cast %addf : tensor<3xf32> to tensor<*xf32>
  call @print_memref_f32(%addf_unranked) : (tensor<*xf32>) -> ()
  // CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [3] strides = [1] data =
  // CHECK-NEXT: [11,  22,  33]

  return
}

func private @print_memref_f32(%ptr : tensor<*xf32>)
