// RUN: mlir-opt --convert-gpu-to-spirv --cse \
// RUN:   --split-input-file --verify-diagnostics %s | FileCheck %s

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.6,
    [Shader, CooperativeMatrixKHR, Float16],
    [SPV_KHR_storage_buffer_storage_class, SPV_KHR_cooperative_matrix]>,
    #spirv.resource_limits<>>} {

  gpu.module @kernels {
    // CHECK-LABEL: spirv.func @gpu_wmma_load_op
    // CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<512 x f32, stride=4> [0])>, StorageBuffer>
    gpu.func @gpu_wmma_load_op(%arg0 : memref<32x32xf16, #spirv.storage_class<StorageBuffer>>) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
      %i = arith.constant 16 : index
      %j = arith.constant 16 : index
      // CHECK:      %[[STRIDE:.+]] = spirv.Constant 32 : i32
      // CHECK:      spirv.KHR.CooperativeMatrixLoad {{%.*}}, %[[STRIDE]], <RowMajor> :
      // CHECK-SAME:   !spirv.ptr<f32, StorageBuffer>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>
      %0 = gpu.subgroup_mma_load_matrix %arg0[%i, %j] {leadDimension = 32 : index} :
        memref<32x32xf16, #spirv.storage_class<StorageBuffer>> -> !gpu.mma_matrix<16x16xf16, "COp">

      // CHECK:      spirv.KHR.CooperativeMatrixLoad {{%.*}}, %[[STRIDE]], <ColumnMajor> :
      // CHECK-SAME:   !spirv.ptr<f32, StorageBuffer>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>
      %1 = gpu.subgroup_mma_load_matrix %arg0[%i, %j] {leadDimension = 32 : index, transpose} :
        memref<32x32xf16, #spirv.storage_class<StorageBuffer>> -> !gpu.mma_matrix<16x16xf16, "COp">
      // CHECK: spirv.Return
      gpu.return
    }

    // CHECK-LABEL: spirv.func @gpu_wmma_store_op
    // CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<512 x f32, stride=4> [0])>, StorageBuffer>
    // CHECK-SAME: !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>
    gpu.func @gpu_wmma_store_op(%arg0: memref<32x32xf16, #spirv.storage_class<StorageBuffer>>,
                                %arg1: !gpu.mma_matrix<16x16xf16, "COp">) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
      %i = arith.constant 16 : index
      %j = arith.constant 16 : index
      // CHECK:      %[[STRIDE:.+]] = spirv.Constant 32 : i32
      // CHECK:      spirv.KHR.CooperativeMatrixStore {{%.*}}, {{%.*}}, %[[STRIDE]], <RowMajor> :
      // CHECK-SAME:  !spirv.ptr<f32, StorageBuffer>, !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>
      gpu.subgroup_mma_store_matrix %arg1, %arg0[%i,%j] {leadDimension = 32 : index} :
        !gpu.mma_matrix<16x16xf16, "COp">, memref<32x32xf16, #spirv.storage_class<StorageBuffer>>

      // CHECK:      spirv.KHR.CooperativeMatrixStore {{%.*}}, {{%.*}}, %[[STRIDE]], <ColumnMajor> :
      // CHECK-SAME:  !spirv.ptr<f32, StorageBuffer>, !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>
      gpu.subgroup_mma_store_matrix %arg1, %arg0[%i,%j] {leadDimension = 32 : index, transpose} :
        !gpu.mma_matrix<16x16xf16, "COp">, memref<32x32xf16, #spirv.storage_class<StorageBuffer>>
       // CHECK: spirv.Return
      gpu.return
    }

    // CHECK-LABEL: spirv.func @gpu_wmma_mma_op
    // CHECK-SAME:    !spirv.coopmatrix<16x16xf16, Subgroup, MatrixA>
    // CHECK-SAME:    !spirv.coopmatrix<16x16xf16, Subgroup, MatrixB>
    // CHECK-SAME:    !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>
    gpu.func @gpu_wmma_mma_op(%A: !gpu.mma_matrix<16x16xf16, "AOp">,
                              %B: !gpu.mma_matrix<16x16xf16, "BOp">,
                              %C: !gpu.mma_matrix<16x16xf16, "COp">,
                              %ptr: memref<16x16xf16, #spirv.storage_class<StorageBuffer>>) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
      // CHECK:      %[[MAD:.*]] = spirv.KHR.CooperativeMatrixMulAdd {{%.*}}, {{%.*}}, {{%.*}} :
      // CHECK-SAME:   !spirv.coopmatrix<16x16xf16, Subgroup, MatrixA>,
      // CHECK-SAME:   !spirv.coopmatrix<16x16xf16, Subgroup, MatrixB>
      // CHECK-SAME:   -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>
      %D = gpu.subgroup_mma_compute %A, %B, %C : !gpu.mma_matrix<16x16xf16, "AOp">,
                                                 !gpu.mma_matrix<16x16xf16, "BOp">
                                                 -> !gpu.mma_matrix<16x16xf16, "COp">

      %i = arith.constant 0 : index
      // CHECK:      spirv.KHR.CooperativeMatrixStore %{{.+}}, %[[MAD]], %{{.+}}, <RowMajor>
      gpu.subgroup_mma_store_matrix %D, %ptr[%i,%i] {leadDimension = 32 : index} :
        !gpu.mma_matrix<16x16xf16, "COp">, memref<16x16xf16, #spirv.storage_class<StorageBuffer>>
      // CHECK: spirv.Return
      gpu.return
    }

    // CHECK-LABEL: spirv.func @gpu_wmma_constant_op
    gpu.func @gpu_wmma_constant_op(%ptr: memref<16x16xf16, #spirv.storage_class<StorageBuffer>>) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
      // CHECK:       %[[CST1F:.+]] = spirv.Constant 1.000000e+00 : f16
      %cst = arith.constant 1.0 : f16
      // CHECK:       %[[MAT:.+]] = spirv.CompositeConstruct %[[CST1F]] :
      // CHECK-SAME:   (f16) -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>
      %C = gpu.subgroup_mma_constant_matrix %cst : !gpu.mma_matrix<16x16xf16, "COp">

      %i = arith.constant 0 : index
      // CHECK:      spirv.KHR.CooperativeMatrixStore %{{.+}}, %[[MAT]], %{{.+}}, <RowMajor>
      gpu.subgroup_mma_store_matrix %C, %ptr[%i,%i] {leadDimension = 32 : index} :
        !gpu.mma_matrix<16x16xf16, "COp">, memref<16x16xf16, #spirv.storage_class<StorageBuffer>>
      // CHECK: spirv.Return
      gpu.return
    }

    // CHECK-LABEL: spirv.func @gpu_wmma_elementwise_op_default
    // CHECK-SAME:    !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>
    // CHECK-SAME:    !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>
    gpu.func @gpu_wmma_elementwise_op_default(%A: !gpu.mma_matrix<16x16xf16, "COp">,
                                              %B: !gpu.mma_matrix<16x16xf16, "COp">,
                                              %ptr: memref<16x16xf32, #spirv.storage_class<StorageBuffer>>) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
      // CHECK:  {{%.*}} = spirv.FAdd {{%.*}}, {{%.*}} : !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>
      %C = gpu.subgroup_mma_elementwise addf %A, %B :
        (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf16, "COp">
      // CHECK:  {{%.*}} = spirv.FNegate {{%.*}} : !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>
      %D = gpu.subgroup_mma_elementwise negatef %C :
        (!gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf16, "COp">
      // CHECK:  {{%.*}} = spirv.FDiv {{%.*}}, {{%.*}} : !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>
      %E = gpu.subgroup_mma_elementwise divf %D, %A :
        (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf16, "COp">
      // CHECK:  {{%.*}} = spirv.FConvert {{%.*}} :
      // CHECK-SAME: !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc> to !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>
      %F = gpu.subgroup_mma_elementwise extf %E :
        (!gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf32, "COp">

      %i = arith.constant 0 : index
      // CHECK: spirv.KHR.CooperativeMatrixStore %{{.+}}, %{{.+}}, %{{.+}}, <RowMajor>
      gpu.subgroup_mma_store_matrix %F, %ptr[%i,%i] {leadDimension = 32 : index} :
        !gpu.mma_matrix<16x16xf32, "COp">, memref<16x16xf32, #spirv.storage_class<StorageBuffer>>
      // CHECK: spirv.Return
      gpu.return
    }

    // CHECK-LABEL: spirv.func @gpu_wmma_elementwise_op_matrix_times_scalar
    // CHECK-SAME:    %[[A:.+]]: !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>
    // CHECK-SAME:    %[[S:.+]]: f16
    gpu.func @gpu_wmma_elementwise_op_matrix_times_scalar(
      %A: !gpu.mma_matrix<16x16xf16, "COp">, %scalar: f16,
      %ptr: memref<16x16xf16, #spirv.storage_class<StorageBuffer>>) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
      %i = arith.constant 0 : index

      %B = gpu.subgroup_mma_constant_matrix %scalar : !gpu.mma_matrix<16x16xf16, "COp">
      // CHECK: %[[C:.+]] = spirv.MatrixTimesScalar %[[A]], %[[S]] : !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>, f16
      // CHECK: spirv.KHR.CooperativeMatrixStore %{{.+}}, %[[C]], %{{.+}}, <RowMajor>
      %C = gpu.subgroup_mma_elementwise mulf %A, %B :
        (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf16, "COp">
      gpu.subgroup_mma_store_matrix %C, %ptr[%i,%i] {leadDimension = 32 : index} :
        !gpu.mma_matrix<16x16xf16, "COp">, memref<16x16xf16, #spirv.storage_class<StorageBuffer>>

      // CHECK: %[[D:.+]] = spirv.MatrixTimesScalar %[[C]], %[[S]] : !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>, f16
      // CHECK: spirv.KHR.CooperativeMatrixStore %{{.+}}, %[[D]], %{{.+}}, <RowMajor>
      %D = gpu.subgroup_mma_elementwise mulf %B, %C :
        (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf16, "COp">
      gpu.subgroup_mma_store_matrix %D, %ptr[%i,%i] {leadDimension = 32 : index} :
        !gpu.mma_matrix<16x16xf16, "COp">, memref<16x16xf16, #spirv.storage_class<StorageBuffer>>
      // CHECK: spirv.Return
      gpu.return
    }

    // CHECK-LABEL: spirv.func @gpu_wmma_elementwise_op_matrix_plus_scalar
    // CHECK-SAME:    %[[A:.+]]: !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>
    // CHECK-SAME:    %[[S:.+]]: f16
    gpu.func @gpu_wmma_elementwise_op_matrix_plus_scalar(
      %A : !gpu.mma_matrix<16x16xf16, "COp">, %scalar : f16,
      %ptr: memref<16x16xf16, #spirv.storage_class<StorageBuffer>>) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
      %i = arith.constant 0 : index

      // CHECK: %[[SM:.+]] = spirv.CompositeConstruct %[[S]] : (f16) -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>
      %B = gpu.subgroup_mma_constant_matrix %scalar : !gpu.mma_matrix<16x16xf16, "COp">
      // CHECK: %[[C:.+]] = spirv.FAdd %[[A]], %[[SM]] : !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>
      %C = gpu.subgroup_mma_elementwise addf %A, %B :
        (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf16, "COp">

      // CHECK: spirv.KHR.CooperativeMatrixStore %{{.+}}, %[[C]], %{{.+}}, <RowMajor>
      gpu.subgroup_mma_store_matrix %C, %ptr[%i,%i] {leadDimension = 32 : index} :
        !gpu.mma_matrix<16x16xf16, "COp">, memref<16x16xf16, #spirv.storage_class<StorageBuffer>>
      // CHECK: spirv.Return
      gpu.return
    }
  }
}
