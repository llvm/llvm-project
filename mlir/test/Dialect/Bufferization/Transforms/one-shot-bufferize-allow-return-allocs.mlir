// RUN: mlir-opt %s -one-shot-bufferize="allow-unknown-ops" -canonicalize -split-input-file | FileCheck %s

// Run fuzzer with different seeds.
// RUN: mlir-opt %s -one-shot-bufferize="test-analysis-only analysis-fuzzer-seed=23" -split-input-file -o /dev/null
// RUN: mlir-opt %s -one-shot-bufferize="test-analysis-only analysis-fuzzer-seed=59" -split-input-file -o /dev/null
// RUN: mlir-opt %s -one-shot-bufferize="test-analysis-only analysis-fuzzer-seed=91" -split-input-file -o /dev/null

// CHECK-LABEL: func @buffer_not_deallocated(
//  CHECK-SAME:     %[[t:.*]]: tensor<?xf32>
func.func @buffer_not_deallocated(%t : tensor<?xf32>, %c : i1) -> tensor<?xf32> {
  // CHECK: %[[m:.*]] = bufferization.to_memref %[[t]]
  // CHECK: %[[r:.*]] = scf.if %{{.*}} {
  %r = scf.if %c -> tensor<?xf32> {
    // CHECK: %[[some_op:.*]] = "test.some_op"
    // CHECK: %[[alloc:.*]] = memref.alloc(%[[some_op]])
    // CHECK: %[[casted:.*]] = memref.cast %[[alloc]]
    // CHECK-NOT: dealloc
    // CHECK: scf.yield %[[casted]]
    %sz = "test.some_op"() : () -> (index)
    %0 = bufferization.alloc_tensor(%sz) : tensor<?xf32>
    scf.yield %0 : tensor<?xf32>
  } else {
  // CHECK: } else {
    // CHECK: scf.yield %[[m]]
    scf.yield %t : tensor<?xf32>
  }
  // CHECK: }
  // CHECK: %[[r_tensor:.*]] = bufferization.to_tensor %[[r]]
  // CHECK: return %[[r_tensor]]
  return %r : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @write_to_alloc_tensor_or_readonly_tensor(
//  CHECK-SAME:     %[[arg0:.*]]: tensor<i32>
func.func @write_to_alloc_tensor_or_readonly_tensor(%arg0: tensor<i32>,
                                                    %cond: i1, %val: i32)
  -> tensor<i32>
{
  // CHECK: %[[arg0_m:.*]] = bufferization.to_memref %[[arg0]]
  // CHECK: %[[r:.*]] = scf.if {{.*}} {
  // CHECK:   scf.yield %[[arg0_m]]
  // CHECK: } else {
  // CHECK:   %[[alloc:.*]] = memref.alloc
  // CHECK:   memref.store %{{.*}}, %[[alloc]]
  // CHECK:   %[[casted:.*]] = memref.cast %[[alloc]]
  // CHECK:   scf.yield %[[casted]]
  // CHECK: }
  // CHECK: %[[r_t:.*]] = bufferization.to_tensor %[[r]]
  // CHECK: return %[[r_t]]
  %3 = scf.if %cond -> (tensor<i32>) {
    scf.yield %arg0 : tensor<i32>
  } else {
    %7 = bufferization.alloc_tensor() : tensor<i32>
    %8 = tensor.insert %val into %7[] : tensor<i32>
    scf.yield %8 : tensor<i32>
  }
  return %3 : tensor<i32>
}
