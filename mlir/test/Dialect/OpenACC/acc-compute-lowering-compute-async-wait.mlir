// RUN: mlir-opt %s -acc-compute-lowering=device-type=nvidia | FileCheck %s --check-prefixes=CHECK,NV
// RUN: mlir-opt %s -acc-compute-lowering | FileCheck %s --check-prefixes=CHECK,NONE

// Device-type-tagged async/wait on compute constructs are lowered to
// acc.kernel_environment with only the clause for the pass device type
// or the default fallback.

// -----

// CHECK-LABEL: func.func @parallel_fallback_none_async_wait
func.func @parallel_fallback_none_async_wait(%buf: memref<1xi32>) {
  %async_none = arith.constant 2 : i32
  %wait_none = arith.constant 1 : i32
  %dev = acc.copyin varPtr(%buf : memref<1xi32>) -> memref<1xi32>
  // CHECK: %[[ASYNC:.*]] = arith.constant 2 : i32
  // CHECK: %[[WAIT:.*]] = arith.constant 1 : i32
  // CHECK-NOT: acc.parallel
  // CHECK: acc.kernel_environment dataOperands({{.*}} : memref<1xi32>)
  // CHECK-SAME: async(%[[ASYNC]] : i32) wait(%[[WAIT]] : i32)
  // CHECK-NOT: wait_devnum
  acc.parallel
      async(%async_none : i32 [#acc.device_type<none>])
      wait({%wait_none : i32} [#acc.device_type<none>])
      dataOperands(%dev : memref<1xi32>) {
    acc.yield
  }
  acc.copyout accPtr(%dev : memref<1xi32>) to varPtr(%buf : memref<1xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @parallel_prefer_nvidia_over_none
func.func @parallel_prefer_nvidia_over_none(%buf: memref<1xi32>) {
  %async_none = arith.constant 1 : i32
  %async_nvidia = arith.constant 2 : i32
  %wait_none = arith.constant 10 : i32
  %wait_nvidia_a = arith.constant 20 : i32
  %wait_nvidia_b = arith.constant 21 : i32
  %dev = acc.copyin varPtr(%buf : memref<1xi32>) -> memref<1xi32>
  // NV: %[[ASYNC_NV:.*]] = arith.constant 2 : i32
  // NV: %[[WAIT_NV_A:.*]] = arith.constant 20 : i32
  // NV: %[[WAIT_NV_B:.*]] = arith.constant 21 : i32
  // NV: acc.kernel_environment dataOperands({{.*}} : memref<1xi32>) async(%[[ASYNC_NV]] : i32) wait(%[[WAIT_NV_A]], %[[WAIT_NV_B]] : i32, i32)
  // NV-NOT: async({{.*}} : i32) wait(%{{[0-9a-z_]+}} : i32)
  // NV-NOT: wait_devnum
  // NONE: %[[ASYNC_NONE:.*]] = arith.constant 1 : i32
  // NONE: %[[WAIT_NONE:.*]] = arith.constant 10 : i32
  // NONE: acc.kernel_environment dataOperands({{.*}} : memref<1xi32>)
  // NONE-SAME: async(%[[ASYNC_NONE]] : i32) wait(%[[WAIT_NONE]] : i32)
  // NONE-NOT: async({{.*}} : i32) wait(%{{.*}}, {{.*}} : i32, i32)
  // NONE-NOT: wait_devnum
  acc.parallel
      async(%async_none : i32 [#acc.device_type<none>],
            %async_nvidia : i32 [#acc.device_type<nvidia>])
      wait({%wait_none : i32} [#acc.device_type<none>],
            {%wait_nvidia_a : i32, %wait_nvidia_b : i32} [#acc.device_type<nvidia>])
      dataOperands(%dev : memref<1xi32>) {
    acc.yield
  }
  acc.copyout accPtr(%dev : memref<1xi32>) to varPtr(%buf : memref<1xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @parallel_nvidia_only_async_wait
func.func @parallel_nvidia_only_async_wait(%buf: memref<1xi32>) {
  %async_nvidia = arith.constant 3 : i32
  %wait_nvidia = arith.constant 30 : i32
  %dev = acc.copyin varPtr(%buf : memref<1xi32>) -> memref<1xi32>
  // NV: %[[ASYNC:.*]] = arith.constant 3 : i32
  // NV: %[[WAIT:.*]] = arith.constant 30 : i32
  // NV: acc.kernel_environment dataOperands({{.*}} : memref<1xi32>)
  // NV-SAME: async(%[[ASYNC]] : i32) wait(%[[WAIT]] : i32)
  // NV-NOT: wait_devnum
  // NONE: acc.kernel_environment dataOperands({{.*}} : memref<1xi32>) {
  // NONE-NOT: async
  // NONE-NOT: wait
  acc.parallel
      async(%async_nvidia : i32 [#acc.device_type<nvidia>])
      wait({%wait_nvidia : i32} [#acc.device_type<nvidia>])
      dataOperands(%dev : memref<1xi32>) {
    acc.yield
  }
  acc.copyout accPtr(%dev : memref<1xi32>) to varPtr(%buf : memref<1xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @parallel_plain_wait_no_devnum
func.func @parallel_plain_wait_no_devnum(%buf: memref<1xi32>) {
  %wait_queue = arith.constant 1 : i32
  %dev = acc.copyin varPtr(%buf : memref<1xi32>) -> memref<1xi32>
  // CHECK: %[[WAIT:.*]] = arith.constant 1 : i32
  // CHECK: acc.kernel_environment dataOperands({{.*}} : memref<1xi32>)
  // CHECK-SAME: wait(%[[WAIT]] : i32)
  // CHECK-NOT: wait_devnum
  acc.parallel wait({%wait_queue : i32} [#acc.device_type<none>])
      dataOperands(%dev : memref<1xi32>) {
    acc.yield
  }
  acc.copyout accPtr(%dev : memref<1xi32>) to varPtr(%buf : memref<1xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @parallel_none_devnum_skipped_for_nvidia_pass
func.func @parallel_none_devnum_skipped_for_nvidia_pass(%buf: memref<1xi32>) {
  %devnum_none = arith.constant 100 : i32
  %wait_none = arith.constant 0 : i32
  %wait_nvidia = arith.constant 4 : i32
  %dev = acc.copyin varPtr(%buf : memref<1xi32>) -> memref<1xi32>
  // NV: %[[WAIT_NV:.*]] = arith.constant 4 : i32
  // NV: acc.kernel_environment dataOperands({{.*}} : memref<1xi32>)
  // NV-SAME: wait(%[[WAIT_NV]] : i32)
  // NV-NOT: wait_devnum
  // NV-NOT: wait_devnum({{.*}} : i32) wait({{.*}} : i32)
  // NONE: %[[DEVNUM_NONE:.*]] = arith.constant 100 : i32
  // NONE: %[[WAIT_NONE:.*]] = arith.constant 0 : i32
  // NONE: acc.kernel_environment dataOperands({{.*}} : memref<1xi32>)
  // NONE-SAME: wait_devnum(%[[DEVNUM_NONE]] : i32) wait(%[[WAIT_NONE]] : i32)
  // NONE-NOT: kernel_environment dataOperands({{.*}} : memref<1xi32>) wait(%{{.*}} : i32) {
  acc.parallel
      wait({devnum: %devnum_none : i32, %wait_none : i32} [#acc.device_type<none>],
            {%wait_nvidia : i32} [#acc.device_type<nvidia>])
      dataOperands(%dev : memref<1xi32>) {
    acc.yield
  }
  acc.copyout accPtr(%dev : memref<1xi32>) to varPtr(%buf : memref<1xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @parallel_nvidia_wait_devnum
func.func @parallel_nvidia_wait_devnum(%buf: memref<1xi32>) {
  %devnum_nvidia = arith.constant 100 : i32
  %wait_nvidia = arith.constant 5 : i32
  %dev = acc.copyin varPtr(%buf : memref<1xi32>) -> memref<1xi32>
  // NV: %[[DEVNUM:.*]] = arith.constant 100 : i32
  // NV: %[[WAIT:.*]] = arith.constant 5 : i32
  // NV: acc.kernel_environment dataOperands({{.*}} : memref<1xi32>)
  // NV-SAME: wait_devnum(%[[DEVNUM]] : i32) wait(%[[WAIT]] : i32)
  // NONE: acc.kernel_environment dataOperands({{.*}} : memref<1xi32>) {
  // NONE-NOT: async
  // NONE-NOT: wait
  acc.parallel
      wait({devnum: %devnum_nvidia : i32, %wait_nvidia : i32}
           [#acc.device_type<nvidia>])
      dataOperands(%dev : memref<1xi32>) {
    acc.yield
  }
  acc.copyout accPtr(%dev : memref<1xi32>) to varPtr(%buf : memref<1xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @kernels_prefer_nvidia_async_wait
func.func @kernels_prefer_nvidia_async_wait(%buf: memref<1xi32>) {
  %async_none = arith.constant 1 : i32
  %async_nvidia = arith.constant 2 : i32
  %wait_none = arith.constant 10 : i32
  %wait_nvidia = arith.constant 20 : i32
  %dev = acc.copyin varPtr(%buf : memref<1xi32>) -> memref<1xi32>
  // CHECK-NOT: acc.kernels
  // NV: %[[ASYNC_NV:.*]] = arith.constant 2 : i32
  // NV: %[[WAIT_NV:.*]] = arith.constant 20 : i32
  // NV: acc.kernel_environment dataOperands({{.*}} : memref<1xi32>)
  // NV-SAME: async(%[[ASYNC_NV]] : i32) wait(%[[WAIT_NV]] : i32)
  // NONE: %[[ASYNC_NONE:.*]] = arith.constant 1 : i32
  // NONE: %[[WAIT_NONE:.*]] = arith.constant 10 : i32
  // NONE: acc.kernel_environment dataOperands({{.*}} : memref<1xi32>)
  // NONE-SAME: async(%[[ASYNC_NONE]] : i32) wait(%[[WAIT_NONE]] : i32)
  acc.kernels
      async(%async_none : i32 [#acc.device_type<none>],
            %async_nvidia : i32 [#acc.device_type<nvidia>])
      wait({%wait_none : i32} [#acc.device_type<none>],
            {%wait_nvidia : i32} [#acc.device_type<nvidia>])
      dataOperands(%dev : memref<1xi32>) {
    acc.terminator
  }
  acc.copyout accPtr(%dev : memref<1xi32>) to varPtr(%buf : memref<1xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @serial_fallback_none_wait
func.func @serial_fallback_none_wait(%buf: memref<1xi32>) {
  %wait_none = arith.constant 7 : i32
  %dev = acc.copyin varPtr(%buf : memref<1xi32>) -> memref<1xi32>
  // CHECK: %[[WAIT:.*]] = arith.constant 7 : i32
  // CHECK-NOT: acc.serial
  // CHECK: acc.kernel_environment dataOperands({{.*}} : memref<1xi32>)
  // CHECK-SAME: wait(%[[WAIT]] : i32)
  // CHECK-NOT: wait_devnum
  acc.serial wait({%wait_none : i32} [#acc.device_type<none>])
      dataOperands(%dev : memref<1xi32>) {
    acc.yield
  }
  acc.copyout accPtr(%dev : memref<1xi32>) to varPtr(%buf : memref<1xi32>)
  return
}
