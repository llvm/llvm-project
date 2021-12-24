// RUN: mlir-opt %s -tensor-constant-bufferize -std-bufferize -linalg-bufferize \
// RUN: -tensor-bufferize -func-bufferize -finalizing-bufferize -buffer-deallocation -convert-linalg-to-loops \
// RUN: -convert-linalg-to-llvm --convert-memref-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

func @foo() -> tensor<4xf32> {
  %0 = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  return %0 : tensor<4xf32>
}

func @main() {
  %0 = call @foo() : () -> tensor<4xf32>

  // Instead of relying on tensor_store which introduces aliasing, we rely on
  // the conversion of print_memref_f32(tensor<*xf32>) to
  // print_memref_f32(memref<*xf32>).
  // Note that this is skipping a step and we would need at least some function
  // attribute to declare that this conversion is valid (e.g. when we statically
  // know that things will play nicely at the C ABI boundary).
  %unranked = tensor.cast %0 : tensor<4xf32> to tensor<*xf32>
  call @print_memref_f32(%unranked) : (tensor<*xf32>) -> ()

  //      CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
  // CHECK-SAME: rank = 1 offset = 0 sizes = [4] strides = [1] data =
  // CHECK-NEXT: [1, 2, 3, 4]

  return
}

// This gets converted to a function operating on memref<*xf32>.
// Note that this is skipping a step and we would need at least some function
// attribute to declare that this conversion is valid (e.g. when we statically
// know that things will play nicely at the C ABI boundary).
func private @print_memref_f32(%ptr : tensor<*xf32>)
