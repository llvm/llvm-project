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

}


