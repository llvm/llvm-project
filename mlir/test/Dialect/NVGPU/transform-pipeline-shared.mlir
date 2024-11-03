// RUN: mlir-opt %s --test-transform-dialect-interpreter --split-input-file --verify-diagnostics | FileCheck %s

func.func @simple_depth_2_unpeeled(%global: memref<?xf32>, %result: memref<?xf32> ) {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c4 = arith.constant 4 : index
  %shared = memref.alloc(%c100) : memref<?xf32, #gpu.address_space<workgroup>>
  %c0f = arith.constant 0.0 : f32
  // Predication is not currently implemented for transfer_read/write, so this is expected to fail.
  // expected-note @below {{couldn't predicate}}
  scf.for %i = %c0 to %c100 step %c4 iter_args(%accum = %c0f) -> f32 {
    %mem = vector.transfer_read %global[%i], %c0f : memref<?xf32>, vector<4xf32>
    vector.transfer_write %mem, %shared[%i] : vector<4xf32>, memref<?xf32, #gpu.address_space<workgroup>>
    %0 = arith.addf %accum, %accum : f32
    scf.yield %0 : f32
  }
  return
}

!t = !transform.any_op

transform.sequence failures(propagate) {
^bb0(%arg0: !t):
  %loop = transform.structured.match ops{["scf.for"]} in %arg0 : (!t) -> !t
  // expected-error @below {{irreversible pipelining failure}}
  // expected-note @below {{try setting "peel_epilogue"}}
  transform.nvgpu.pipeline_shared_memory_copies failures(propagate) %loop { depth = 2 } : (!t) -> !t
}

// -----

// Loop pipeliner is tested separately, just verify the overall shape of the IR here.

func.func private @body(index, memref<?xf32, #gpu.address_space<workgroup>>)

// CHECK-LABEL: @simple_depth_2_peeled
// CHECK-SAME: %[[ARG:.+]]: memref
func.func @simple_depth_2_peeled(%global: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %c4 = arith.constant 4 : index
  // CHECK: memref.alloc
  %shared = memref.alloc(%c200) : memref<?xf32, #gpu.address_space<workgroup>>
  %c0f = arith.constant 0.0 : f32
  // CHECK: %[[LOADED1:.+]] = vector.transfer_read %[[ARG]]
  // CHECK: %[[LOADED2:.+]] = vector.transfer_read %[[ARG]]
  // CHECK: %[[LOOP:.+]]:2 = scf.for {{.*}} iter_args(%[[IA1:.+]] = %[[LOADED1]], %[[IA2:.+]] = %[[LOADED2]])
  // CHECK:   vector.transfer_write %[[IA1]]
  // CHECK:   func.call @body
  // CHECK:   %[[LOCAL_LOADED:.+]] = vector.transfer_read %[[ARG]]
  // CHECK:   scf.yield %[[IA2]], %[[LOCAL_LOADED]]
  scf.for %i = %c0 to %c100 step %c4 {
    %mem = vector.transfer_read %global[%i], %c0f : memref<?xf32>, vector<4xf32>
    vector.transfer_write %mem, %shared[%i] : vector<4xf32>, memref<?xf32, #gpu.address_space<workgroup>>
    func.call @body(%i, %shared) : (index, memref<?xf32, #gpu.address_space<workgroup>>) -> ()
  }
  // CHECK: vector.transfer_write %[[LOOP]]#0
  // CHECK: call @body
  // CHECK: vector.transfer_write %[[LOOP]]#1
  // CHECK: call @body
  return
}

!t = !transform.any_op

transform.sequence failures(propagate) {
^bb0(%arg0: !t):
  %loop = transform.structured.match ops{["scf.for"]} in %arg0 : (!t) -> !t
  transform.nvgpu.pipeline_shared_memory_copies failures(propagate) %loop { depth = 2, peel_epilogue } : (!t) -> !t
}

// -----

// CHECK-LABEL: @async_depth_2_predicated
// CHECK-SAME: %[[GLOBAL:.+]]: memref
func.func @async_depth_2_predicated(%global: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c98 = arith.constant 98 : index
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK: %[[C4:.+]] = arith.constant 4
  %c4 = arith.constant 4 : index
  // CHECK: %[[SHARED:.+]] = memref.alloc{{.*}} #gpu.address_space<workgroup>
  %shared = memref.alloc(%c200) : memref<?xf32, #gpu.address_space<workgroup>>
  %c0f = arith.constant 0.0 : f32
  // CHECK: %[[TOKEN0:.+]] = nvgpu.device_async_copy
  // CHECK: %[[TOKEN1:.+]] = nvgpu.device_async_copy
  // CHECK: scf.for %[[I:.+]] = {{.*}} iter_args
  // CHECK-SAME: %[[ITER_ARG0:.+]] = %[[TOKEN0]]
  // CHECK-SAME: %[[ITER_ARG1:.+]] = %[[TOKEN1]]
  scf.for %i = %c0 to %c98 step %c4 {
    // Condition for the predication "select" below.
    // CHECK:   %[[C90:.+]] = arith.constant 90
    // CHECK:   %[[CMP0:.+]] = arith.cmpi slt, %[[I]], %[[C90]]
    // CHECK:   nvgpu.device_async_wait %[[ITER_ARG0]] {numGroups = 1

    // Original "select" with updated induction variable.
    // CHECK:   %[[C96:.+]] = arith.constant 96
    // CHECK:   %[[C8:.+]] = arith.constant 8
    // CHECK:   %[[I_PLUS_8:.+]] = arith.addi %[[I]], %[[C8]]
    // CHECK:   %[[CMP1:.+]] = arith.cmpi slt, %[[I_PLUS_8]], %[[C96]]
    // CHECK:   %[[C2:.+]] = arith.constant 2
    // CHECK:   %[[SELECTED0:.+]] = arith.select %[[CMP1]], %[[C4]], %[[C2]]
    %c96 = arith.constant 96 : index
    %cond = arith.cmpi slt, %i, %c96 : index
    %c2 = arith.constant 2 : index
    %read_size = arith.select %cond, %c4, %c2 : index

    // Updated induction variables (two more) for the device_async_copy below.
    // These are generated repeatedly by the pipeliner.
    // CHECK:   %[[C8_2:.+]] = arith.constant 8
    // CHECK:   %[[I_PLUS_8_2:.+]] = arith.addi %[[I]], %[[C8_2]]
    // CHECK:   %[[C8_3:.+]] = arith.constant 8
    // CHECK:   %[[I_PLUS_8_3:.+]] = arith.addi %[[I]], %[[C8_3]]

    // The second "select" is generated by predication and selects 0 for
    // the two last iterations.
    // CHECK:   %[[C0:.+]] = arith.constant 0
    // CHECK:   %[[SELECTED1:.+]] = arith.select %[[CMP0]], %[[SELECTED0]], %[[C0]]
    // CHECK:   %[[ASYNC_TOKEN:.+]] = nvgpu.device_async_copy %[[GLOBAL]][%[[I_PLUS_8_3]]], %[[SHARED]][%[[I_PLUS_8_2]]], 4, %[[SELECTED1]]
    %token = nvgpu.device_async_copy %global[%i], %shared[%i], 4, %read_size
      : memref<?xf32> to memref<?xf32, #gpu.address_space<workgroup>>

    nvgpu.device_async_wait %token

    // CHECK: scf.yield %[[ITER_ARG1]], %[[ASYNC_TOKEN]]
  }
  // There is no need to wait for the last copies as it it was fully predicated
  // out and doesn't load the original data.
  // CHECK-NOT: nvgpu.device_async_wait
  return
}


!t = !transform.any_op

transform.sequence failures(propagate) {
^bb0(%arg0: !t):
  %loop = transform.structured.match ops{["scf.for"]} in %arg0 : (!t) -> !t
  transform.nvgpu.pipeline_shared_memory_copies failures(propagate) %loop { depth = 2 } : (!t) -> !t
}

// -----

// CHECK-LABEL: @async_depth_2_peeled
func.func @async_depth_2_peeled(%global: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c98 = arith.constant 98 : index
  %c100 = arith.constant 100 : index
  %c4 = arith.constant 4 : index
  %shared = memref.alloc(%c100) : memref<?xf32, #gpu.address_space<workgroup>>
  %c0f = arith.constant 0.0 : f32
  // CHECK: nvgpu.device_async_copy
  // CHECK: nvgpu.device_async_copy
  // CHECK: scf.for
  // CHECK:   nvgpu.device_async_wait %{{.*}} {numGroups = 1
  // CHECK:   arith.select
  // CHECK:   nvgpu.device_async_copy
  // CHECK:   scf.yield
  // CHECK: nvgpu.device_async_wait %{{.*}} {numGroups = 1
  // CHEKC: nvgpu.device_async_wait %{{.*}} {numGroups = 0
  scf.for %i = %c0 to %c98 step %c4 {
    %c96 = arith.constant 96 : index
    %cond = arith.cmpi slt, %i, %c96 : index
    %c2 = arith.constant 2 : index
    %read_size = arith.select %cond, %c4, %c2 : index
    %token = nvgpu.device_async_copy %global[%i], %shared[%i], 4, %read_size
      : memref<?xf32> to memref<?xf32, #gpu.address_space<workgroup>>
    nvgpu.device_async_wait %token
  }
  return
}


!t = !transform.any_op

transform.sequence failures(propagate) {
^bb0(%arg0: !t):
  %loop = transform.structured.match ops{["scf.for"]} in %arg0 : (!t) -> !t
  transform.nvgpu.pipeline_shared_memory_copies failures(propagate) %loop { depth = 2, peel_epilogue } : (!t) -> !t
}
