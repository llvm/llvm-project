// RUN: mlir-opt -gpu-async-region %s | FileCheck %s

// CHECK: module attributes {gpu.container_module}
module attributes {gpu.container_module} {

  gpu.module @kernels {
    gpu.func @kernel() kernel { gpu.return }
  }

  func private @foo() -> ()

  // CHECK-LABEL:func @async(%{{.*}}: index)
  func @async(%sz : index) {
    // CHECK: %[[t0:.*]] = gpu.wait async
    // CHECK: %[[t1:.*]] = gpu.launch_func async [%[[t0]]]
    gpu.launch_func @kernels::@kernel
        blocks in (%sz, %sz, %sz) threads in (%sz, %sz, %sz)
    // CHECK: %[[t2:.*]] = gpu.launch_func async [%[[t1]]]
    gpu.launch_func @kernels::@kernel
        blocks in (%sz, %sz, %sz) threads in (%sz, %sz, %sz)
    // CHECK: gpu.wait [%[[t2]]]
    // CHECK: call @foo
    call @foo() : () -> ()
    return
  }

  // CHECK-LABEL:func @defer_wait(%{{.*}}: index)
  func @defer_wait(%sz : index) {
    // CHECK: %[[a0:.*]], %[[f0:.*]] = async.execute
    %a0 = async.execute {
      // CHECK: %[[t:.*]] = gpu.launch_func async
      gpu.launch_func @kernels::@kernel
          blocks in (%sz, %sz, %sz) threads in (%sz, %sz, %sz)
      // CHECK-NOT: gpu.wait
      // CHECK: async.yield %[[t]]
      async.yield
    }

    // CHECK: %[[a1:.*]], %[[f1:.*]] = async.execute
    // CHECK-SAME: %[[f0]]
    %a1 = async.execute [%a0] {
      // CHECK: %[[t:.*]] = gpu.launch_func async
      gpu.launch_func @kernels::@kernel
          blocks in (%sz, %sz, %sz) threads in (%sz, %sz, %sz)
      // CHECK-NOT: gpu.wait
      // CHECK: async.yield %[[t]]
      async.yield
    }

    // CHECK: async.await %[[a1]]
    // CHECK: %[[t:.*]] = async.await %[[f1]]
    // CHECK: gpu.wait [%[[t]]]
    async.await %a1 : !async.token
    return
  }

  // CHECK-LABEL:func @defer_wait_blocked_by_side_effect(%{{.*}}: index)
  func @defer_wait_blocked_by_side_effect(%sz : index) {
    // CHECK: %[[a:.*]] = async.execute
    %a = async.execute {
      // CHECK: %[[t:.*]] = gpu.launch_func async
      gpu.launch_func @kernels::@kernel
          blocks in (%sz, %sz, %sz) threads in (%sz, %sz, %sz)
      // CHECK: gpu.wait [%[[t]]]
      call @foo() : () -> ()
      async.yield
    }

    // CHECK: async.await %[[a]]
    // CHECK-NOT: gpu.wait
    async.await %a : !async.token
    return
  }

  // CHECK-LABEL:func @defer_wait_pass_through(%{{.*}}: index)
  func @defer_wait_pass_through(%sz : index) {
    // CHECK: %[[a0:.*]], %[[f0:.*]] = async.execute
    %a0 = async.execute {
      // CHECK: %[[t:.*]] = gpu.launch_func async
      gpu.launch_func @kernels::@kernel
          blocks in (%sz, %sz, %sz) threads in (%sz, %sz, %sz)
      // CHECK-NOT: gpu.wait
      // CHECK: async.yield %[[t]]
      async.yield
    }

    // CHECK: %[[a1:.*]], %[[f1:.*]] = async.execute
    // CHECK-SAME: %[[f0]]
    %a1 = async.execute [%a0] {
      // CHECK-NOT: gpu.wait
      // CHECK: async.yield %{{.*}}
      async.yield
    }

    // CHECK: async.await %[[a1]]
    // CHECK: %[[t:.*]] = async.await %[[f1]]
    // CHECK: gpu.wait [%[[t]]]
    async.await %a1 : !async.token
    return
  }
}
