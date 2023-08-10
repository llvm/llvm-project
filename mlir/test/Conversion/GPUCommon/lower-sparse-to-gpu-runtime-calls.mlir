// RUN: mlir-opt %s --gpu-to-llvm='use-opaque-pointers=1' | FileCheck %s

module attributes {gpu.container_module} {

  // CHECK-LABEL: func @matvec
  // CHECK: llvm.call @mgpuStreamCreate
  // CHECK: llvm.call @mgpuMemAlloc
  // CHECK: llvm.call @mgpuMemAlloc
  // CHECK: llvm.call @mgpuCreateCoo
  // CHECK: llvm.call @mgpuCreateDnVec
  // CHECK: llvm.call @mgpuSpMVBufferSize
  // CHECK: llvm.call @mgpuSpMV
  // CHECK: llvm.call @mgpuDestroySpMat
  // CHECK: llvm.call @mgpuDestroyDnVec
  // CHECK: llvm.call @mgpuStreamSynchronize
  // CHECK: llvm.call @mgpuStreamDestroy
  func.func @matvec(%arg0: index) {
    %token0 = gpu.wait async
    %mem1, %token1 = gpu.alloc async [%token0] (%arg0) : memref<?xindex>
    %mem2, %token2 = gpu.alloc async [%token1] (%arg0) : memref<?xf64>
    %spmat, %token4 = gpu.create_coo async [%token2] %arg0, %arg0, %arg0, %mem1, %mem1, %mem2 : memref<?xindex>, memref<?xindex>, memref<?xf64>
    %dnvec, %token5 = gpu.create_dn_tensor async [%token4] %mem2, %arg0 : index into memref<?xf64>
    %bufferSz, %token6 = gpu.spmv_buffer_size async [%token5] %spmat, %dnvec, %dnvec  into f64
    %token7 = gpu.spmv async [%token6] %spmat, %dnvec, %dnvec, %mem2 : memref<?xf64> into f64
    %token8 = gpu.destroy_sp_mat async [%token7] %spmat
    %token9 = gpu.destroy_dn_tensor async [%token8] %dnvec
    gpu.wait [%token9]
    return
  }

  // CHECK-LABEL: func @matmul
  // CHECK: llvm.call @mgpuStreamCreate
  // CHECK: llvm.call @mgpuMemAlloc
  // CHECK: llvm.call @mgpuMemAlloc
  // CHECK: llvm.call @mgpuCreateCsr
  // CHECK: llvm.call @mgpuCreateDnMat
  // CHECK: llvm.call @mgpuSpMMBufferSize
  // CHECK: llvm.call @mgpuSpMM
  // CHECK: llvm.call @mgpuDestroySpMat
  // CHECK: llvm.call @mgpuDestroyDnMat
  // CHECK: llvm.call @mgpuStreamSynchronize
  // CHECK: llvm.call @mgpuStreamDestroy
  func.func @matmul(%arg0: index) {
    %token0 = gpu.wait async
    %mem1, %token1 = gpu.alloc async [%token0] (%arg0) : memref<?xindex>
    %mem2, %token2 = gpu.alloc async [%token1] (%arg0) : memref<?xf64>
    %spmat, %token4 = gpu.create_csr async [%token2] %arg0, %arg0, %arg0, %mem1, %mem1, %mem2 : memref<?xindex>, memref<?xindex>, memref<?xf64>
    %dnmat, %token5 = gpu.create_dn_tensor async [%token4] %mem2, %arg0, %arg0 : index, index into memref<?xf64>
    %bufferSz, %token6 = gpu.spmm_buffer_size async [%token5] %spmat, %dnmat, %dnmat : index into f64
    %token7 = gpu.spmm async [%token6] %spmat, %dnmat, %dnmat, %mem2 : memref<?xf64> into f64
    %token8 = gpu.destroy_sp_mat async [%token7] %spmat
    %token9 = gpu.destroy_dn_tensor async [%token8] %dnmat
    gpu.wait [%token9]
    return
  }

  // CHECK-LABEL: func @sddmm
  // CHECK: llvm.call @mgpuStreamCreate
  // CHECK: llvm.call @mgpuMemAlloc
  // CHECK: llvm.call @mgpuMemAlloc
  // CHECK: llvm.call @mgpuCreateCsr
  // CHECK: llvm.call @mgpuCreateDnMat
  // CHECK: llvm.call @mgpuSDDMMBufferSize
  // CHECK: llvm.call @mgpuSDDMM
  // CHECK: llvm.call @mgpuDestroySpMat
  // CHECK: llvm.call @mgpuDestroyDnMat
  // CHECK: llvm.call @mgpuStreamSynchronize
  // CHECK: llvm.call @mgpuStreamDestroy
  func.func @sddmm(%arg0: index) {
    %token0 = gpu.wait async
    %mem1, %token1 = gpu.alloc async [%token0] (%arg0) : memref<?xindex>
    %mem2, %token2 = gpu.alloc async [%token1] (%arg0) : memref<?xf64>
    %spmat, %token4 = gpu.create_csr async [%token2] %arg0, %arg0, %arg0, %mem1, %mem1, %mem2 : memref<?xindex>, memref<?xindex>, memref<?xf64>
    %dnmat, %token5 = gpu.create_dn_tensor async [%token4] %mem2, %arg0, %arg0 : index, index into memref<?xf64>
    %bufferSz, %token6 = gpu.sddmm_buffer_size async [%token5] %dnmat, %dnmat, %spmat into f64
    %token7 = gpu.sddmm async [%token6]  %dnmat, %dnmat, %spmat, %mem2 : memref<?xf64> into f64
    %token8 = gpu.destroy_sp_mat async [%token7] %spmat
    %token9 = gpu.destroy_dn_tensor async [%token8] %dnmat
    gpu.wait [%token9]
    return
  }


  // CHECK-LABEL:     func @spgemm
  // CHECK: llvm.call @mgpuStreamCreate
  // CHECK: llvm.call @mgpuMemAlloc
  // CHECK: llvm.call @mgpuMemAlloc
  // CHECK: llvm.call @mgpuCreateCsr
  // CHECK: llvm.call @mgpuCreateCsr
  // CHECK: llvm.call @mgpuCreateCsr
  // CHECK: llvm.call @mgpuSpGEMMCreateDescr
  // CHECK: llvm.call @malloc
  // CHECK: llvm.call @mgpuSpGEMMWorkEstimation
  // CHECK: llvm.call @mgpuMemAlloc
  // CHECK: llvm.call @mgpuSpGEMMWorkEstimation
  // CHECK: llvm.call @mgpuMemAlloc
  // CHECK: llvm.call @mgpuSpGEMMCompute
  // CHECK: llvm.call @mgpuMemAlloc
  // CHECK: llvm.call @mgpuMemAlloc
  // CHECK: llvm.call @mgpuStreamSynchronize
  // CHECK: llvm.call @mgpuStreamDestroy
  // CHECK: llvm.call @mgpuStreamCreate
  // CHECK: llvm.call @mgpuSpGEMMCopy
  // CHECK: llvm.call @mgpuDestroySpMat
  // CHECK: llvm.call @mgpuDestroySpMat
  // CHECK: llvm.call @mgpuDestroySpMat
  // CHECK: llvm.call @mgpuStreamSynchronize
  // CHECK: llvm.call @mgpuStreamDestroy
  func.func @spgemm(%arg0: index) {
    %token0 = gpu.wait async
    %mem1, %token1 = gpu.alloc async [%token0] (%arg0) : memref<?xindex>
    %mem2, %token2 = gpu.alloc async [%token1] (%arg0) : memref<?xf64>
    %spmatA, %token3 = gpu.create_csr async [%token2] %arg0, %arg0, %arg0, %mem1, %mem1, %mem2 : memref<?xindex>, memref<?xindex>, memref<?xf64>
    %spmatB, %token4 = gpu.create_csr async [%token3] %arg0, %arg0, %arg0, %mem1, %mem1, %mem2 : memref<?xindex>, memref<?xindex>, memref<?xf64>
    %spmatC, %token5 = gpu.create_csr async [%token4] %arg0, %arg0, %arg0, %mem1, %mem1, %mem2 : memref<?xindex>, memref<?xindex>, memref<?xf64>
    %spgemmDesc, %token6 = gpu.spgemm_create_descr async [%token5]
    // Used as nullptr
    %alloc = memref.alloc() : memref<0xi8>
    %c0 = arith.constant 0 : index
    %bufferSz1, %token7 = gpu.spgemm_work_estimation_or_compute async
                            [%token6]{WORK_ESTIMATION}
                            %spmatA{NON_TRANSPOSE}, %spmatB{NON_TRANSPOSE},
                            %spmatC, %spgemmDesc, %c0,
                            %alloc: f32 into memref<0xi8>
    %buf1, %token8 = gpu.alloc async [%token7] (%bufferSz1) : memref<?xi8>
    %bufferSz1_1, %token9 = gpu.spgemm_work_estimation_or_compute async
                              [%token8]{WORK_ESTIMATION} %spmatA, %spmatB,
                              %spmatC, %spgemmDesc, %bufferSz1,
                              %buf1: f32 into memref<?xi8>
    %buf2, %token13 = gpu.alloc async [%token9] (%bufferSz1_1) : memref<?xi8>
    %bufferSz2_2, %token14 = gpu.spgemm_work_estimation_or_compute async
                               [%token13]{COMPUTE} %spmatA, %spmatB, %spmatC,
                               %spgemmDesc, %bufferSz1_1,
                               %buf2: f32 into memref<?xi8>
    %rows, %cols, %nnz, %token15 = gpu.spgemm_get_size async [%token14] %spmatC
    %mem_columns, %token16 = gpu.alloc async [%token15] (%cols) : memref<?xi32>
    %mem_values, %token17 = gpu.alloc async [%token16] (%nnz) : memref<?xf32>
    gpu.wait [%token17]
    %token18 = gpu.wait async
    %token19 = gpu.spgemm_copy async [%token18] %spmatA, %spmatB, %spmatC, %spgemmDesc: f32
    %token20 = gpu.destroy_sp_mat async [%token19] %spmatA
    %token21 = gpu.destroy_sp_mat async [%token20] %spmatB
    %token22 = gpu.destroy_sp_mat async [%token21] %spmatC
    gpu.wait [%token22]
    return
  }

}
