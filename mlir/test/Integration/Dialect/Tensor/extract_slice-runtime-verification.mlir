// RUN: mlir-opt %s -generate-runtime-verification \
// RUN:     -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -buffer-deallocation-pipeline=private-function-dynamic-ownership \
// RUN:     -test-cf-assert \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-to-llvm | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%tlir_runner_utils 2>&1 | \
// RUN: FileCheck %s

// RUN: mlir-opt %s -generate-runtime-verification \
// RUN:     -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -buffer-deallocation-pipeline=private-function-dynamic-ownership \
// RUN:     -test-cf-assert \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-to-llvm="allow-pattern-rollback=0" \
// RUN:     -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%tlir_runner_utils 2>&1 | \
// RUN: FileCheck %s

func.func @extract_slice(%tensor: tensor<1xf32>, %offset: index) {
    tensor.extract_slice %tensor[%offset] [1] [1] : tensor<1xf32> to tensor<1xf32>
    return
}

func.func @extract_slice_dynamic(%tensor: tensor<?x4xf32>, %offset: index, %size: index, %stride: index) {
    tensor.extract_slice %tensor[%offset, 0] [%size, 4] [%stride, 1] : tensor<?x4xf32> to tensor<?x4xf32>
    return
}

func.func @extract_slice_dynamic_rank_reduce(%tensor: tensor<?x4xf32>, %offset: index, %size: index, %stride: index) {
    tensor.extract_slice %tensor[%offset, 0] [%size, 1] [%stride, 1] : tensor<?x4xf32> to tensor<?xf32>
    return
}

func.func @extract_slice_zero_size_dim(%arg0: tensor<10x4x1xf32>, %dim_0: index, %dim_1: index, %dim_2: index) {
    tensor.extract_slice %arg0[0, 0, 0] [%dim_0, %dim_1, %dim_2] [1, 1, 1] : tensor<10x4x1xf32> to tensor<?x?x?xf32>
    return
}


func.func @main() {
  %0 = arith.constant 0 : index
  %1 = arith.constant 1 : index
  %n1 = arith.constant -1 : index
  %4 = arith.constant 4 : index
  %5 = arith.constant 5 : index

  %alloca = tensor.empty() : tensor<1xf32>
  %alloca_4 = tensor.empty() : tensor<4x4xf32>
  %alloca_4_dyn = tensor.cast %alloca_4 : tensor<4x4xf32> to tensor<?x4xf32>

  // Offset is out-of-bounds and slice runs out-of-bounds
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: tensor.extract_slice %{{.*}}[%{{.*}}, 0] [%{{.*}}, 1] [%{{.*}}, 1] : tensor<?x4xf32> to tensor<?xf32>
  // CHECK-NEXT: ^ offset 0 is out-of-bounds
  // CHECK-NEXT: Location: loc({{.*}})
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: tensor.extract_slice %{{.*}}[%{{.*}}, 0] [%{{.*}}, 1] [%{{.*}}, 1] : tensor<?x4xf32> to tensor<?xf32>
  // CHECK-NEXT: ^ extract_slice runs out-of-bounds along dimension 0
  // CHECK-NEXT: Location: loc({{.*}})
  func.call @extract_slice_dynamic_rank_reduce(%alloca_4_dyn, %5, %5, %1) : (tensor<?x4xf32>, index, index, index) -> ()

  // Offset is out-of-bounds and slice runs out-of-bounds
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: tensor.extract_slice %{{.*}}[%{{.*}}] [1] [1] : tensor<1xf32> to tensor<1xf32>
  // CHECK-NEXT: ^ offset 0 is out-of-bounds
  // CHECK-NEXT: Location: loc({{.*}})
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: tensor.extract_slice %{{.*}}[%{{.*}}] [1] [1] : tensor<1xf32> to tensor<1xf32>
  // CHECK-NEXT: ^ extract_slice runs out-of-bounds along dimension 0
  // CHECK-NEXT: Location: loc({{.*}})
  func.call @extract_slice(%alloca, %1) : (tensor<1xf32>, index) -> ()

  // Offset is out-of-bounds and slice runs out-of-bounds
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: tensor.extract_slice %{{.*}}[%{{.*}}] [1] [1] : tensor<1xf32> to tensor<1xf32>
  // CHECK-NEXT: ^ offset 0 is out-of-bounds
  // CHECK-NEXT: Location: loc({{.*}})
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: tensor.extract_slice %{{.*}}[%{{.*}}] [1] [1] : tensor<1xf32> to tensor<1xf32>
  // CHECK-NEXT: ^ extract_slice runs out-of-bounds along dimension 0
  // CHECK-NEXT: Location: loc({{.*}})
  func.call @extract_slice(%alloca, %n1) : (tensor<1xf32>, index) -> ()

  // Slice runs out-of-bounds due to size
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: tensor.extract_slice %{{.*}}[%{{.*}}, 0] [%{{.*}}, 4] [%{{.*}}, 1] : tensor<?x4xf32> to tensor<?x4xf32>
  // CHECK-NEXT: ^ extract_slice runs out-of-bounds along dimension 0
  // CHECK-NEXT: Location: loc({{.*}})
  func.call @extract_slice_dynamic(%alloca_4_dyn, %0, %5, %1) : (tensor<?x4xf32>, index, index, index) -> ()

  // Slice runs out-of-bounds due to stride
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: tensor.extract_slice %{{.*}}[%{{.*}}, 0] [%{{.*}}, 4] [%{{.*}}, 1] : tensor<?x4xf32> to tensor<?x4xf32>
  // CHECK-NEXT: ^ extract_slice runs out-of-bounds along dimension 0
  // CHECK-NEXT: Location: loc({{.*}})
  func.call @extract_slice_dynamic(%alloca_4_dyn, %0, %4, %4) : (tensor<?x4xf32>, index, index, index) -> ()

  // CHECK-NOT: ERROR: Runtime op verification failed
  func.call @extract_slice(%alloca, %0) : (tensor<1xf32>, index) -> ()

  // CHECK-NOT: ERROR: Runtime op verification failed
  func.call @extract_slice_dynamic(%alloca_4_dyn, %0, %4, %1) : (tensor<?x4xf32>, index, index, index) -> ()

  // CHECK-NOT: ERROR: Runtime op verification failed
  func.call @extract_slice_dynamic_rank_reduce(%alloca_4_dyn, %0, %1, %0) : (tensor<?x4xf32>, index, index, index) -> ()

  %cst10x4x1xf32 = arith.constant dense<1.0> : tensor<10x4x1xf32>
  
  // CHECK-NOT: ERROR: Runtime op verification failed
  %dim_0 = arith.constant 0 : index
  %dim_1 = arith.constant 4 : index
  %dim_2 = arith.constant 1 : index
  func.call @extract_slice_zero_size_dim(%cst10x4x1xf32, %dim_0, %dim_1, %dim_2) : (tensor<10x4x1xf32>, index, index, index) -> ()

  return
}
