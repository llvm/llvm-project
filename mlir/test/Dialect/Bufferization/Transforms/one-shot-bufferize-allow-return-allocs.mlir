// RUN: mlir-opt %s -one-shot-bufferize="allow-return-allocs allow-unknown-ops" -split-input-file | FileCheck %s

// Run fuzzer with different seeds.
// RUN: mlir-opt %s -one-shot-bufferize="allow-return-allocs test-analysis-only analysis-fuzzer-seed=23" -split-input-file -o /dev/null
// RUN: mlir-opt %s -one-shot-bufferize="allow-return-allocs test-analysis-only analysis-fuzzer-seed=59" -split-input-file -o /dev/null
// RUN: mlir-opt %s -one-shot-bufferize="allow-return-allocs test-analysis-only analysis-fuzzer-seed=91" -split-input-file -o /dev/null

// CHECK-LABEL: func @buffer_not_deallocated(
//  CHECK-SAME:     %[[t:.*]]: tensor<?xf32>
func.func @buffer_not_deallocated(%t : tensor<?xf32>, %c : i1) -> tensor<?xf32> {
  // CHECK: %[[r:.*]] = scf.if %{{.*}} {
  %r = scf.if %c -> tensor<?xf32> {
    // CHECK: %[[some_op:.*]] = "test.some_op"
    // CHECK: %[[alloc:.*]] = memref.alloc(%[[some_op]])
    // CHECK: %[[casted:.*]] = memref.cast %[[alloc]]
    // CHECK-NOT: dealloc
    // CHECK: scf.yield %[[casted]]
    %sz = "test.some_op"() : () -> (index)
    %0 = bufferization.alloc_tensor[%sz] : tensor<?xf32>
    scf.yield %0 : tensor<?xf32>
  } else {
  // CHECK: } else {
    // CHECK: %[[m:.*]] = bufferization.to_memref %[[t]]
    // CHECK: %[[cloned:.*]] = bufferization.clone %[[m]]
    // CHECK: scf.yield %[[cloned]]
    scf.yield %t : tensor<?xf32>
  }
  // CHECK: }
  // CHECK: %[[r_tensor:.*]] = bufferization.to_tensor %[[r]]
  // CHECK: memref.dealloc %[[r]]
  // CHECK: return %[[r_tensor]]
  return %r : tensor<?xf32>
}
