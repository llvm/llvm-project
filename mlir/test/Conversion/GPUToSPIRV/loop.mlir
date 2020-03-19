// RUN: mlir-opt -convert-gpu-to-spirv %s -o - | FileCheck %s

module attributes {
  gpu.container_module,
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>
} {
  func @loop(%arg0 : memref<10xf32>, %arg1 : memref<10xf32>) {
    %c0 = constant 1 : index
    "gpu.launch_func"(%c0, %c0, %c0, %c0, %c0, %c0, %arg0, %arg1) { kernel = "loop_kernel", kernel_module = @kernels} : (index, index, index, index, index, index, memref<10xf32>, memref<10xf32>) -> ()
    return
  }

  gpu.module @kernels {
    gpu.func @loop_kernel(%arg2 : memref<10xf32>, %arg3 : memref<10xf32>)
    attributes {gpu.kernel, spv.entry_point_abi = {local_size = dense<[16, 1, 1]>: vector<3xi32>}} {
      // CHECK: [[LB:%.*]] = spv.constant 4 : i32
      %lb = constant 4 : index
      // CHECK: [[UB:%.*]] = spv.constant 42 : i32
      %ub = constant 42 : index
      // CHECK: [[STEP:%.*]] = spv.constant 2 : i32
      %step = constant 2 : index
      // CHECK:      spv.loop {
      // CHECK-NEXT:   spv.Branch [[HEADER:\^.*]]([[LB]] : i32)
      // CHECK:      [[HEADER]]([[INDVAR:%.*]]: i32):
      // CHECK:        [[CMP:%.*]] = spv.SLessThan [[INDVAR]], [[UB]] : i32
      // CHECK:        spv.BranchConditional [[CMP]], [[BODY:\^.*]], [[MERGE:\^.*]]
      // CHECK:      [[BODY]]:
      // CHECK:        [[STRIDE1:%.*]] = spv.constant 1 : i32
      // CHECK:        [[OFFSET1:%.*]] = spv.IMul [[STRIDE1]], [[INDVAR]] : i32
      // CHECK:        spv.AccessChain {{%.*}}{{\[}}{{%.*}}, [[OFFSET1]]{{\]}} : {{.*}}
      // CHECK:        [[STRIDE2:%.*]] = spv.constant 1 : i32
      // CHECK:        [[OFFSET2:%.*]] = spv.IMul [[STRIDE2]], [[INDVAR]] : i32
      // CHECK:        spv.AccessChain {{%.*}}{{\[}}{{%.*}}, [[OFFSET2]]{{\]}} : {{.*}}
      // CHECK:        [[INCREMENT:%.*]] = spv.IAdd [[INDVAR]], [[STEP]] : i32
      // CHECK:        spv.Branch [[HEADER]]([[INCREMENT]] : i32)
      // CHECK:      [[MERGE]]
      // CHECK:        spv._merge
      // CHECK:      }
      loop.for %arg4 = %lb to %ub step %step {
        %1 = load %arg2[%arg4] : memref<10xf32>
        store %1, %arg3[%arg4] : memref<10xf32>
      }
      gpu.return
    }
  }
}
