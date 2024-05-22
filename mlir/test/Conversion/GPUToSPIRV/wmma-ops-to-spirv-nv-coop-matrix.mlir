// RUN: mlir-opt --convert-gpu-to-spirv="use-coop-matrix-nv=true" \
// RUN:   --split-input-file --verify-diagnostics %s | FileCheck %s

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, CooperativeMatrixNV, Float16], [SPV_KHR_storage_buffer_storage_class, SPV_NV_cooperative_matrix]>, #spirv.resource_limits<>>} {
  gpu.module @kernels {
    // CHECK-LABEL: spirv.func @gpu_wmma_load_op
    // CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<512 x f32, stride=4> [0])>, StorageBuffer>
    gpu.func @gpu_wmma_load_op(%arg0 : memref<32x32xf16, #spirv.storage_class<StorageBuffer>>) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
      %i = arith.constant 16 : index
      %j = arith.constant 16 : index
      // CHECK: %[[COLMAJOR:.*]] = spirv.Constant false
      // CHECK: {{%.*}} = spirv.NV.CooperativeMatrixLoad {{%.*}}, {{%.*}}, %[[COLMAJOR]] :  !spirv.ptr<f32, StorageBuffer> as !spirv.NV.coopmatrix<16x16xf16, Subgroup>
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
    // CHECK-LABEL: spirv.func @gpu_wmma_load_op_transpose
    // CHECK-SAME: {{%.*}}: !spirv.ptr<!spirv.struct<(!spirv.array<512 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>}
    // CHECK-SAME: spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>
    gpu.func @gpu_wmma_load_op_transpose(%arg0 : memref<32x32xf16, #spirv.storage_class<StorageBuffer>>) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
      %i = arith.constant 16 : index
      %j = arith.constant 16 : index
      // CHECK: %[[COLMAJOR:.*]] = spirv.Constant true
      // CHECK: {{%.*}} = spirv.NV.CooperativeMatrixLoad {{%.*}}, {{%.*}}, %[[COLMAJOR]] :  !spirv.ptr<f32, StorageBuffer> as !spirv.NV.coopmatrix<16x16xf16, Subgroup>
      %0 = gpu.subgroup_mma_load_matrix %arg0[%i, %j] {leadDimension = 32 : index, transpose} : memref<32x32xf16, #spirv.storage_class<StorageBuffer>> -> !gpu.mma_matrix<16x16xf16, "COp">
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
    // CHECK-LABEL: spirv.func @gpu_wmma_store_op
    // CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<512 x f32, stride=4> [0])>, StorageBuffer>
    // CHECK-SAME: !spirv.NV.coopmatrix<16x16xf16, Subgroup>
    gpu.func @gpu_wmma_store_op(%arg0 : memref<32x32xf16, #spirv.storage_class<StorageBuffer>>, %arg1 : !gpu.mma_matrix<16x16xf16, "COp">) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
      %i = arith.constant 16 : index
      %j = arith.constant 16 : index
      // CHECK: %[[COLMAJOR:.*]] = spirv.Constant false
      //  CHECK: spirv.NV.CooperativeMatrixStore {{%.*}}, {{%.*}}, {{%.*}}, %[[COLMAJOR]] : !spirv.ptr<f32, StorageBuffer>, !spirv.NV.coopmatrix<16x16xf16, Subgroup>
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
    // CHECK-LABEL: spirv.func @gpu_wmma_store_op_transpose
    // CHECK-SAME: {{%.*}}: !spirv.ptr<!spirv.struct<(!spirv.array<512 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>}
    // CHECK-SAME: {{%.*}}: !spirv.NV.coopmatrix<16x16xf16, Subgroup> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>})
    // CHECK-SAME: spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>
    gpu.func @gpu_wmma_store_op_transpose(%arg0 : memref<32x32xf16, #spirv.storage_class<StorageBuffer>>, %arg1 : !gpu.mma_matrix<16x16xf16, "COp">) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
      %i = arith.constant 16 : index
      %j = arith.constant 16 : index
      // CHECK: %[[COLMAJOR:.*]] = spirv.Constant true
      // CHECK: spirv.NV.CooperativeMatrixStore {{%.*}}, {{%.*}}, {{%.*}}, %[[COLMAJOR]] : !spirv.ptr<f32, StorageBuffer>, !spirv.NV.coopmatrix<16x16xf16, Subgroup>
      gpu.subgroup_mma_store_matrix %arg1, %arg0[%i,%j] {leadDimension= 32 : index, transpose} : !gpu.mma_matrix<16x16xf16, "COp">, memref<32x32xf16,  #spirv.storage_class<StorageBuffer>>
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
    // CHECK-LABEL: spirv.func @gpu_wmma_mma_op
    // CHECK-SAME: !spirv.NV.coopmatrix<16x16xf16, Subgroup>
    // CHECK-SAME: !spirv.NV.coopmatrix<16x16xf16, Subgroup>
    // CHECK-SAME: !spirv.NV.coopmatrix<16x16xf16, Subgroup>
    gpu.func @gpu_wmma_mma_op(%A : !gpu.mma_matrix<16x16xf16, "AOp">, %B : !gpu.mma_matrix<16x16xf16, "BOp">, %C : !gpu.mma_matrix<16x16xf16, "COp">) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
      // CHECK: {{%.*}} = spirv.NV.CooperativeMatrixMulAdd {{%.*}}, {{%.*}}, {{%.*}} : !spirv.NV.coopmatrix<16x16xf16, Subgroup>, !spirv.NV.coopmatrix<16x16xf16, Subgroup> -> !spirv.NV.coopmatrix<16x16xf16, Subgroup>
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
    // CHECK-LABEL: spirv.func @gpu_wmma_constant_op
    gpu.func @gpu_wmma_constant_op() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
      // CHECK: {{%.*}} = spirv.Constant
      %cst = arith.constant 1.0 : f16
      // CHECK: {{%.*}} = spirv.CompositeConstruct {{%.*}} : (f16) -> !spirv.NV.coopmatrix<16x16xf16, Subgroup>
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
    // CHECK-LABEL: spirv.func @gpu_wmma_elementwise_op_default
    // CHECK-SAME: !spirv.NV.coopmatrix<16x16xf16, Subgroup>
    // CHECK-SAME: !spirv.NV.coopmatrix<16x16xf16, Subgroup>
    gpu.func @gpu_wmma_elementwise_op_default(%A : !gpu.mma_matrix<16x16xf16, "COp">, %B : !gpu.mma_matrix<16x16xf16, "COp">) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
      // CHECK:  {{%.*}} = spirv.FAdd {{%.*}}, {{%.*}} : !spirv.NV.coopmatrix<16x16xf16, Subgroup>
      %C = gpu.subgroup_mma_elementwise addf %A, %B : (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf16, "COp">
      // CHECK:  {{%.*}} = spirv.FNegate {{%.*}} : !spirv.NV.coopmatrix<16x16xf16, Subgroup>
      %D = gpu.subgroup_mma_elementwise negatef %C : (!gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf16, "COp">
      // CHECK:  {{%.*}} = spirv.FDiv {{%.*}}, {{%.*}} : !spirv.NV.coopmatrix<16x16xf16, Subgroup>
      %E = gpu.subgroup_mma_elementwise divf %D, %A : (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf16, "COp">
      // CHECK:  {{%.*}} = spirv.FConvert {{%.*}} : !spirv.NV.coopmatrix<16x16xf16, Subgroup> to !spirv.NV.coopmatrix<16x16xf32, Subgroup>
      %F = gpu.subgroup_mma_elementwise extf %E : (!gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf32, "COp">
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
    // CHECK-LABEL: spirv.func @gpu_wmma_elementwise_op_matrix_times_scalar
    // CHECK-SAME:    %[[A:.+]]: !spirv.NV.coopmatrix<16x16xf16, Subgroup>
    // CHECK-SAME:    %[[S:.+]]: f16
    gpu.func @gpu_wmma_elementwise_op_matrix_times_scalar(%A : !gpu.mma_matrix<16x16xf16, "COp">, %scalar : f16) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
      %B = gpu.subgroup_mma_constant_matrix %scalar : !gpu.mma_matrix<16x16xf16, "COp">
      // CHECK: %{{.+}} = spirv.MatrixTimesScalar %[[A]], %[[S]] : !spirv.NV.coopmatrix<16x16xf16, Subgroup>, f16
      %C = gpu.subgroup_mma_elementwise mulf %A, %B : (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf16, "COp">
      // CHECK: %{{.+}} = spirv.MatrixTimesScalar %[[A]], %[[S]] : !spirv.NV.coopmatrix<16x16xf16, Subgroup>, f16
      %D = gpu.subgroup_mma_elementwise mulf %B, %A : (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf16, "COp">
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
    // CHECK-LABEL: spirv.func @gpu_wmma_elementwise_op_matrix_plus_scalar
    // CHECK-SAME:    %[[A:.+]]: !spirv.NV.coopmatrix<16x16xf16, Subgroup>
    // CHECK-SAME:    %[[S:.+]]: f16
    gpu.func @gpu_wmma_elementwise_op_matrix_plus_scalar(%A : !gpu.mma_matrix<16x16xf16, "COp">, %scalar : f16) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
      // CHECK: %[[SM:.+]] = spirv.CompositeConstruct %[[S]] : (f16) -> !spirv.NV.coopmatrix<16x16xf16, Subgroup>
      %B = gpu.subgroup_mma_constant_matrix %scalar : !gpu.mma_matrix<16x16xf16, "COp">
      // CHECK: %{{.+}} = spirv.FAdd %[[A]], %[[SM]] : !spirv.NV.coopmatrix<16x16xf16, Subgroup>
      %C = gpu.subgroup_mma_elementwise addf %A, %B : (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf16, "COp">
      gpu.return
    }
  }
}
