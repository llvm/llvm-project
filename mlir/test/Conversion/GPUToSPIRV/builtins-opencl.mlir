// RUN: mlir-opt -split-input-file -convert-gpu-to-spirv="use-64bit-index=false" %s -o - | FileCheck %s --check-prefix=INDEX32
// RUN: mlir-opt -split-input-file -convert-gpu-to-spirv="use-64bit-index=true" %s -o - | FileCheck %s --check-prefix=INDEX64

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Kernel, Int64], []>, #spirv.resource_limits<>>
} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_workgroup_id_x
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // INDEX32-LABEL:  spirv.module @{{.*}} Physical32 OpenCL
  // INDEX32: spirv.GlobalVariable [[WORKGROUPID:@.*]] built_in("WorkgroupId") : !spirv.ptr<vector<3xi32>, Input>
  // INDEX64-LABEL:  spirv.module @{{.*}} Physical64 OpenCL
  // INDEX64: spirv.GlobalVariable [[WORKGROUPID:@.*]] built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
  gpu.module @kernels {
    gpu.func @builtin_workgroup_id_x() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      // INDEX32: [[ADDRESS:%.*]] = spirv.mlir.addressof [[WORKGROUPID]]
      // INDEX32-NEXT: [[VEC:%.*]] = spirv.Load "Input" [[ADDRESS]]
      // INDEX32-NEXT: {{%.*}} = spirv.CompositeExtract [[VEC]]{{\[}}0 : i32{{\]}}
      // INDEX64-NOT: spirv.UConvert
      %0 = gpu.block_id x
      gpu.return
    }
  }
}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Kernel, Int64], []>, #spirv.resource_limits<>>
} {
  // INDEX32-LABEL:  spirv.module @{{.*}} Physical32 OpenCL
  // INDEX32: spirv.GlobalVariable [[SUBGROUPSIZE:@.*]] built_in("SubgroupSize") : !spirv.ptr<i32, Input>
  // INDEX64-LABEL:  spirv.module @{{.*}} Physical64 OpenCL
  // INDEX64: spirv.GlobalVariable [[SUBGROUPSIZE:@.*]] built_in("SubgroupSize") : !spirv.ptr<i32, Input>
  gpu.module @kernels {
    gpu.func @builtin_subgroup_size() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      // INDEX32: [[ADDRESS:%.*]] = spirv.mlir.addressof [[SUBGROUPSIZE]]
      // INDEX32-NEXT: {{%.*}} = spirv.Load "Input" [[ADDRESS]]
      // INDEX64: spirv.UConvert %{{.+}} : i32 to i64
      %0 = gpu.subgroup_size : index
      gpu.return
    }
  }
}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Kernel, Int64], []>, #spirv.resource_limits<>>
} {
  // INDEX32-LABEL:  spirv.module @{{.*}} Physical32 OpenCL
  // INDEX32: spirv.GlobalVariable [[LANEID:@.*]] built_in("SubgroupLocalInvocationId") : !spirv.ptr<i32, Input>
  // INDEX64-LABEL:  spirv.module @{{.*}} Physical64 OpenCL
  // INDEX64: spirv.GlobalVariable [[LANEID:@.*]] built_in("SubgroupLocalInvocationId") : !spirv.ptr<i32, Input>
  gpu.module @kernels {
    gpu.func @builtin_laneid() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      // INDEX32: [[ADDRESS:%.*]] = spirv.mlir.addressof [[LANEID]]
      // INDEX32-NEXT: {{%.*}} = spirv.Load "Input" [[ADDRESS]]
      // INDEX64: spirv.UConvert %{{.+}} : i32 to i64
      %0 = gpu.lane_id
      gpu.return
    }
  }
}
