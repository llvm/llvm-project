// RUN: mlir-opt %s --gpu-to-llvm | FileCheck %s

module attributes {gpu.container_module} {

  gpu.module @foo {
    gpu.func @bar() kernel {
      gpu.return
    }
  }

  // CHECK-LABEL: func @main
  func.func @main() {
    %c1 = arith.constant 1 : index

    // Check that pass does not modify launch_func ops with only 1 dependency:

    // CHECK: llvm.call @mgpuStreamCreate() : () -> !llvm.ptr
    // CHECK: llvm.call @mgpuEventCreate() : () -> !llvm.ptr
    // CHECK: llvm.call @mgpuEventRecord(%{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr) -> ()
    %t0 = gpu.wait async
    // CHECK: llvm.call @mgpuEventCreate() : () -> !llvm.ptr
    // CHECK: llvm.call @mgpuEventRecord(%{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr) -> ()
    // CHECK: llvm.call @mgpuStreamCreate() : () -> !llvm.ptr
    // CHECK: llvm.call @mgpuStreamWaitEvent(%{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr) -> ()
    // CHECK: llvm.call @mgpuEventDestroy(%{{.*}}) : (!llvm.ptr) -> ()
    %t1 = gpu.wait async [%t0]
    // CHECK: llvm.call @mgpuEventCreate() : () -> !llvm.ptr
    // CHECK: llvm.call @mgpuEventRecord(%{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr) -> ()
    // CHECK: llvm.call @mgpuStreamCreate() : () -> !llvm.ptr
    // CHECK: llvm.call @mgpuStreamWaitEvent(%{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr) -> ()
    // CHECK: llvm.call @mgpuStreamWaitEvent(%{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr) -> ()
    // CHECK: llvm.call @mgpuEventDestroy(%{{.*}}) : (!llvm.ptr) -> ()
    // CHECK: llvm.call @mgpuEventDestroy(%{{.*}}) : (!llvm.ptr) -> ()
    %0 = gpu.wait async [%t0, %t1]
    // CHECK: gpu.launch_func <%{{.*}} : !llvm.ptr> @foo::@bar blocks in (%{{.*}}, %{{.*}}, %{{.*}}) threads in (%{{.*}}, %{{.*}}, %{{.*}}) : i64
    %good_call = gpu.launch_func async [%0] @foo::@bar
        blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
    // CHECK: llvm.call @mgpuStreamSynchronize(%{{.*}}) : (!llvm.ptr) -> ()
    // CHECK: llvm.call @mgpuStreamDestroy(%{{.*}}) : (!llvm.ptr) -> ()
    gpu.wait [%good_call]

    // Check that launch_func ops with multiple dependencies are properly
    // handled and do not result in a failure:

    // CHECK: llvm.call @mgpuStreamCreate() : () -> !llvm.ptr
    // CHECK: llvm.call @mgpuEventCreate() : () -> !llvm.ptr
    // CHECK: llvm.call @mgpuEventRecord(%{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr) -> ()
    %t2 = gpu.wait async
    // CHECK: llvm.call @mgpuEventCreate() : () -> !llvm.ptr
    // CHECK: llvm.call @mgpuEventRecord(%{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr) -> ()
    // CHECK: llvm.call @mgpuStreamCreate() : () -> !llvm.ptr
    // CHECK: llvm.call @mgpuStreamWaitEvent(%{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr) -> ()
    // CHECK: llvm.call @mgpuEventDestroy(%{{.*}}) : (!llvm.ptr) -> ()
    %t3 = gpu.wait async [%t2]
    // Inserted gpu.wait:
    // CHECK: llvm.call @mgpuEventCreate() : () -> !llvm.ptr
    // CHECK: llvm.call @mgpuEventRecord(%{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr) -> ()
    // CHECK: llvm.call @mgpuStreamCreate() : () -> !llvm.ptr
    // CHECK: llvm.call @mgpuStreamWaitEvent(%{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr) -> ()
    // CHECK: llvm.call @mgpuStreamWaitEvent(%{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr) -> ()
    // CHECK: llvm.call @mgpuEventDestroy(%{{.*}}) : (!llvm.ptr) -> ()
    // CHECK: llvm.call @mgpuEventDestroy(%{{.*}}) : (!llvm.ptr) -> ()
    // gpu.launch_func only has 1 async dependency:
    // CHECK: gpu.launch_func <%{{.*}} : !llvm.ptr> @foo::@bar blocks in (%{{.*}}, %{{.*}}, %{{.*}}) threads in (%{{.*}}, %{{.*}}, %{{.*}}) : i64
    %bad_call = gpu.launch_func async [%t2, %t3] @foo::@bar
        blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
    // CHECK: llvm.call @mgpuStreamSynchronize(%{{.*}}) : (!llvm.ptr) -> ()
    // CHECK: llvm.call @mgpuStreamDestroy(%{{.*}}) : (!llvm.ptr) -> ()
    gpu.wait [%bad_call]
    return
  }

  // func.func @foo(%size : index) -> memref<?xf32> {
  //   %t0 = gpu.wait async
  //   %t1 = gpu.wait async [%t0]
  //   %0 = gpu.alloc [%t0, %t1] (%size) : memref<?xf32>
  //   // gpu.wait [%1]
  //   return %0 : memref<?xf32>
  // }

}
