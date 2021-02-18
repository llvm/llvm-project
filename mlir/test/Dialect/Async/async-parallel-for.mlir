// RUN: mlir-opt %s -async-parallel-for | FileCheck %s

// CHECK-LABEL: @loop_1d
func @loop_1d(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<?xf32>) {
  // CHECK: %[[GROUP:.*]] = async.create_group
  // CHECK: scf.for
  // CHECK:   %[[TOKEN:.*]] = async.execute {
  // CHECK:     scf.for
  // CHECK:       memref.store
  // CHECK:     async.yield
  // CHECK:   }
  // CHECK:   async.add_to_group %[[TOKEN]], %[[GROUP]]
  // CHECK: async.await_all %[[GROUP]]
  scf.parallel (%i) = (%arg0) to (%arg1) step (%arg2) {
    %one = constant 1.0 : f32
    memref.store %one, %arg3[%i] : memref<?xf32>
  }

  return
}

// CHECK-LABEL: @loop_2d
func @loop_2d(%arg0: index, %arg1: index, %arg2: index, // lb, ub, step
              %arg3: index, %arg4: index, %arg5: index, // lb, ub, step
              %arg6: memref<?x?xf32>) {
  // CHECK: %[[GROUP:.*]] = async.create_group
  // CHECK: scf.for
  // CHECK:   scf.for
  // CHECK:     %[[TOKEN:.*]] = async.execute {
  // CHECK:       scf.for
  // CHECK:         scf.for
  // CHECK:           memref.store
  // CHECK:       async.yield
  // CHECK:     }
  // CHECK:     async.add_to_group %[[TOKEN]], %[[GROUP]]
  // CHECK: async.await_all %[[GROUP]]
  scf.parallel (%i0, %i1) = (%arg0, %arg3) to (%arg1, %arg4)
                            step (%arg2, %arg5) {
    %one = constant 1.0 : f32
    memref.store %one, %arg6[%i0, %i1] : memref<?x?xf32>
  }

  return
}
