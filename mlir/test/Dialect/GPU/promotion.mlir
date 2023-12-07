// RUN: mlir-opt -allow-unregistered-dialect -pass-pipeline='builtin.module(gpu.module(gpu.func(test-gpu-memory-promotion)))' -split-input-file %s | FileCheck %s

gpu.module @foo {

  // Verify that the attribution was indeed introduced
  // CHECK-LABEL: @memref3d
  // CHECK-SAME: (%[[arg:.*]]: memref<5x4xf32>
  // CHECK-SAME: workgroup(%[[promoted:.*]] : memref<5x4xf32, #gpu.address_space<workgroup>>)
  gpu.func @memref3d(%arg0: memref<5x4xf32> {gpu.test_promote_workgroup}) kernel {
    // Verify that loop bounds are emitted, the order does not matter.
    // CHECK-DAG: %[[c1:.*]] = arith.constant 1
    // CHECK-DAG: %[[c4:.*]] = arith.constant 4
    // CHECK-DAG: %[[c5:.*]] = arith.constant 5
    // CHECK-DAG: %[[tx:.*]] = gpu.thread_id x
    // CHECK-DAG: %[[ty:.*]] = gpu.thread_id y
    // CHECK-DAG: %[[tz:.*]] = gpu.thread_id z
    // CHECK-DAG: %[[bdx:.*]] = gpu.block_dim x
    // CHECK-DAG: %[[bdy:.*]] = gpu.block_dim y
    // CHECK-DAG: %[[bdz:.*]] = gpu.block_dim z

    // Verify that loops for the copy are emitted. We only check the number of
    // loops here since their bounds are produced by mapLoopToProcessorIds,
    // tested separately.
    // CHECK: scf.for %[[i0:.*]] =
    // CHECK:   scf.for %[[i1:.*]] =
    // CHECK:     scf.for %[[i2:.*]] =

    // Verify that the copy is emitted and uses only the last two loops.
    // CHECK:       %[[v:.*]] = memref.load %[[arg]][%[[i1]], %[[i2]]]
    // CHECK:       store %[[v]], %[[promoted]][%[[i1]], %[[i2]]]

    // Verify that the use has been rewritten.
    // CHECK: "use"(%[[promoted]]) : (memref<5x4xf32, #gpu.address_space<workgroup>>)
    "use"(%arg0) : (memref<5x4xf32>) -> ()


    // Verify that loops for the copy are emitted. We only check the number of
    // loops here since their bounds are produced by mapLoopToProcessorIds,
    // tested separately.
    // CHECK: scf.for %[[i0:.*]] =
    // CHECK:   scf.for %[[i1:.*]] =
    // CHECK:     scf.for %[[i2:.*]] =

    // Verify that the copy is emitted and uses only the last two loops.
    // CHECK:       %[[v:.*]] = memref.load %[[promoted]][%[[i1]], %[[i2]]]
    // CHECK:       store %[[v]], %[[arg]][%[[i1]], %[[i2]]]
    gpu.return
  }
}

// -----

gpu.module @foo {

  // Verify that the attribution was indeed introduced
  // CHECK-LABEL: @memref5d
  // CHECK-SAME: (%[[arg:.*]]: memref<8x7x6x5x4xf32>
  // CHECK-SAME: workgroup(%[[promoted:.*]] : memref<8x7x6x5x4xf32, #gpu.address_space<workgroup>>)
  gpu.func @memref5d(%arg0: memref<8x7x6x5x4xf32> {gpu.test_promote_workgroup}) kernel {
    // Verify that loop bounds are emitted, the order does not matter.
    // CHECK-DAG: %[[c0:.*]] = arith.constant 0
    // CHECK-DAG: %[[c1:.*]] = arith.constant 1
    // CHECK-DAG: %[[c4:.*]] = arith.constant 4
    // CHECK-DAG: %[[c5:.*]] = arith.constant 5
    // CHECK-DAG: %[[c6:.*]] = arith.constant 6
    // CHECK-DAG: %[[c7:.*]] = arith.constant 7
    // CHECK-DAG: %[[c8:.*]] = arith.constant 8
    // CHECK-DAG: %[[tx:.*]] = gpu.thread_id x
    // CHECK-DAG: %[[ty:.*]] = gpu.thread_id y
    // CHECK-DAG: %[[tz:.*]] = gpu.thread_id z
    // CHECK-DAG: %[[bdx:.*]] = gpu.block_dim x
    // CHECK-DAG: %[[bdy:.*]] = gpu.block_dim y
    // CHECK-DAG: %[[bdz:.*]] = gpu.block_dim z

    // Verify that loops for the copy are emitted.
    // CHECK: scf.for %[[i0:.*]] =
    // CHECK:   scf.for %[[i1:.*]] =
    // CHECK:     scf.for %[[i2:.*]] =
    // CHECK:       scf.for %[[i3:.*]] =
    // CHECK:         scf.for %[[i4:.*]] =

    // Verify that the copy is emitted.
    // CHECK:           %[[v:.*]] = memref.load %[[arg]][%[[i0]], %[[i1]], %[[i2]], %[[i3]], %[[i4]]]
    // CHECK:           store %[[v]], %[[promoted]][%[[i0]], %[[i1]], %[[i2]], %[[i3]], %[[i4]]]

    // Verify that the use has been rewritten.
    // CHECK: "use"(%[[promoted]]) : (memref<8x7x6x5x4xf32, #gpu.address_space<workgroup>>)
    "use"(%arg0) : (memref<8x7x6x5x4xf32>) -> ()

    // Verify that loop loops for the copy are emitted.
    // CHECK: scf.for %[[i0:.*]] =
    // CHECK:   scf.for %[[i1:.*]] =
    // CHECK:     scf.for %[[i2:.*]] =
    // CHECK:       scf.for %[[i3:.*]] =
    // CHECK:         scf.for %[[i4:.*]] =

    // Verify that the copy is emitted.
    // CHECK:           %[[v:.*]] = memref.load %[[promoted]][%[[i0]], %[[i1]], %[[i2]], %[[i3]], %[[i4]]]
    // CHECK:           store %[[v]], %[[arg]][%[[i0]], %[[i1]], %[[i2]], %[[i3]], %[[i4]]]
    gpu.return
  }
}

// -----

gpu.module @foo {

  // Check that attribution insertion works fine.
  // CHECK-LABEL: @insert
  // CHECK-SAME: (%{{.*}}: memref<4xf32>
  // CHECK-SAME: workgroup(%{{.*}}: memref<1x1xf64, #gpu.address_space<workgroup>>
  // CHECK-SAME: %[[wg2:.*]] : memref<4xf32, #gpu.address_space<workgroup>>)
  // CHECK-SAME: private(%{{.*}}: memref<1x1xi64, 5>)
  gpu.func @insert(%arg0: memref<4xf32> {gpu.test_promote_workgroup})
      workgroup(%arg1: memref<1x1xf64, #gpu.address_space<workgroup>>)
      private(%arg2: memref<1x1xi64, 5>)
      kernel {
    // CHECK: "use"(%[[wg2]])
    "use"(%arg0) : (memref<4xf32>) -> ()
    gpu.return
  }
}
