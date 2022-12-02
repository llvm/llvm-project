// RUN: mlir-opt -allow-unregistered-dialect -convert-gpu-to-spirv -split-input-file -verify-diagnostics %s | FileCheck %s

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, CooperativeMatrixNV, Float16], [SPV_KHR_storage_buffer_storage_class, SPV_NV_cooperative_matrix]>, #spirv.resource_limits<>>} {
  gpu.module @kernels {
    // CHECK:       spirv.module @{{.*}} Logical GLSL450 {
    // CHECK-LABEL: spirv.func @gpu_wmma_load_op
    // CHECK-SAME: {{%.*}}: !spirv.ptr<!spirv.struct<(!spirv.array<512 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>}
    // CHECK-SAME: spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>
    gpu.func @gpu_wmma_load_op(%arg0 : memref<32x32xf16, #spirv.storage_class<StorageBuffer>>) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
      %i = arith.constant 16 : index
      %j = arith.constant 16 : index
      // CHECK: {{%.*}} = spirv.NV.CooperativeMatrixLoad {{%.*}}, {{%.*}}, {{%.*}} :  !spirv.ptr<f32, StorageBuffer> as !spirv.coopmatrix<16x16xf16, Subgroup>
      %0 = gpu.subgroup_mma_load_matrix %arg0[%i, %j] {leadDimension = 32 : index} : memref<32x32xf16, #spirv.storage_class<StorageBuffer>> -> !gpu.mma_matrix<16x16xf16, "COp">
      // CHECK: spirv.Return
      gpu.return
    }
  }
}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, CooperativeMatrixNV, Float16], [SPV_KHR_storage_buffer_storage_class, SPV_NV_cooperative_matrix]>, #spirv.resource_limits<>>} {
  gpu.module @kernels {
    // CHECK:       spirv.module @{{.*}} Logical GLSL450 {
    // CHECK-LABEL: spirv.func @gpu_wmma_store_op
    // CHECK-SAME: {{%.*}}: !spirv.ptr<!spirv.struct<(!spirv.array<512 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>}
    // CHECK-SAME: {{%.*}}: !spirv.coopmatrix<16x16xf16, Subgroup> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>})
    // CHECK-SAME: spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>
    gpu.func @gpu_wmma_store_op(%arg0 : memref<32x32xf16, #spirv.storage_class<StorageBuffer>>, %arg1 : !gpu.mma_matrix<16x16xf16, "COp">) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
      %i = arith.constant 16 : index
      %j = arith.constant 16 : index
      //  CHECK: spirv.NV.CooperativeMatrixStore {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}} : !spirv.ptr<f32, StorageBuffer>, !spirv.coopmatrix<16x16xf16, Subgroup>
      gpu.subgroup_mma_store_matrix %arg1, %arg0[%i,%j] {leadDimension= 32 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<32x32xf16,  #spirv.storage_class<StorageBuffer>>
      // CHECK: spirv.Return
      gpu.return
    }
  }
}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, CooperativeMatrixNV, Float16], [SPV_KHR_storage_buffer_storage_class, SPV_NV_cooperative_matrix]>, #spirv.resource_limits<>>} {
  gpu.module @kernels {
    // CHECK:       spirv.module @{{.*}} Logical GLSL450 {
    // CHECK-LABEL: spirv.func @gpu_wmma_mma_op
    // CHECK-SAME: {{%.*}}: !spirv.coopmatrix<16x16xf16, Subgroup> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>}
    // CHECK-SAME: {{%.*}}: !spirv.coopmatrix<16x16xf16, Subgroup> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>}
    // CHECK-SAME: {{%.*}}: !spirv.coopmatrix<16x16xf16, Subgroup> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 2)>})
    // CHECK-SAME: spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>
    gpu.func @gpu_wmma_mma_op(%A : !gpu.mma_matrix<16x16xf16, "AOp">, %B : !gpu.mma_matrix<16x16xf16, "BOp">, %C : !gpu.mma_matrix<16x16xf16, "COp">) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
      // CHECK: {{%.*}} = spirv.NV.CooperativeMatrixMulAdd {{%.*}}, {{%.*}}, {{%.*}} : !spirv.coopmatrix<16x16xf16, Subgroup>, !spirv.coopmatrix<16x16xf16, Subgroup> -> !spirv.coopmatrix<16x16xf16, Subgroup>
      %D = gpu.subgroup_mma_compute %A, %B, %C : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
      // CHECK: spirv.Return
      gpu.return
    }
  }
}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, CooperativeMatrixNV, Float16], [SPV_KHR_storage_buffer_storage_class, SPV_NV_cooperative_matrix]>, #spirv.resource_limits<>>} {
  gpu.module @kernels {
    // CHECK:       spirv.module @{{.*}} Logical GLSL450 {
    // CHECK-LABEL: spirv.func @gpu_wmma_constant_op
    gpu.func @gpu_wmma_constant_op() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
      // CHECK: {{%.*}} = spirv.Constant
      %cst = arith.constant 1.0 : f16
      // CHECK: {{%.*}} = spirv.CompositeConstruct {{%.*}} : (f16) -> !spirv.coopmatrix<16x16xf16, Subgroup>
      %C = gpu.subgroup_mma_constant_matrix %cst : !gpu.mma_matrix<16x16xf16, "COp">
      // CHECK: spirv.Return
      gpu.return
    }
  }
}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, CooperativeMatrixNV, Float16], [SPV_KHR_storage_buffer_storage_class, SPV_NV_cooperative_matrix]>, #spirv.resource_limits<>>} {
  gpu.module @kernels {
    // CHECK:       spirv.module @{{.*}} Logical GLSL450 {
    // CHECK-LABEL: spirv.func @gpu_wmma_elementwise_op
    // CHECK-SAME: {{%.*}}: !spirv.coopmatrix<16x16xf16, Subgroup> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>}
    // CHECK-SAME: {{%.*}}: !spirv.coopmatrix<16x16xf16, Subgroup> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>})
    gpu.func @gpu_wmma_elementwise_op(%A : !gpu.mma_matrix<16x16xf16, "COp">, %B : !gpu.mma_matrix<16x16xf16, "COp">) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
      // CHECK:  {{%.*}} = spirv.FAdd {{%.*}}, {{%.*}} : !spirv.coopmatrix<16x16xf16, Subgroup>
      %C = gpu.subgroup_mma_elementwise addf %A, %B : (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf16, "COp">
      // CHECK:  {{%.*}} = spirv.FNegate {{%.*}} : !spirv.coopmatrix<16x16xf16, Subgroup>
      %D = gpu.subgroup_mma_elementwise negatef %C : (!gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf16, "COp">
      // CHECK:  {{%.*}} = spirv.FDiv {{%.*}}, {{%.*}} : !spirv.coopmatrix<16x16xf16, Subgroup>
      %E = gpu.subgroup_mma_elementwise divf %D, %A : (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf16, "COp">
      // CHECK: spirv.Return
      gpu.return
    }
  }
}