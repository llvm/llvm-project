// RUN: mlir-opt %s -linalg-bufferize -std-bufferize -tensor-constant-bufferize -func-bufferize \
// RUN: -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

func @main() {
  %const = constant dense<10.0> : tensor<2xf32>
  %insert_val = constant dense<20.0> : tensor<1xf32>

  // Both of these subtensor_insert ops insert into the same original tensor
  // value `%const`. This can easily cause bugs if at the memref level
  // we attempt to write in-place into the memref that %const has been
  // converted into.
  %inserted_at_position_0 = subtensor_insert %insert_val into %const[0][1][1] : tensor<1xf32> into tensor<2xf32>
  %inserted_at_position_1 = subtensor_insert %insert_val into %const[1][1][1] : tensor<1xf32> into tensor<2xf32>

  %unranked_at_position_0 = tensor_cast %inserted_at_position_0 : tensor<2xf32> to tensor<*xf32>
  call @print_memref_f32(%unranked_at_position_0) : (tensor<*xf32>) -> ()

  //      CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
  // CHECK-SAME: rank = 1 offset = 0 sizes = [2] strides = [1] data =
  // CHECK-NEXT: [20, 10]

  %unranked_at_position_1 = tensor_cast %inserted_at_position_1 : tensor<2xf32> to tensor<*xf32>
  call @print_memref_f32(%unranked_at_position_1) : (tensor<*xf32>) -> ()

  //      CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
  // CHECK-SAME: rank = 1 offset = 0 sizes = [2] strides = [1] data =
  // CHECK-NEXT: [10, 20]

  return
}

func private @print_memref_f32(%ptr : tensor<*xf32>)
