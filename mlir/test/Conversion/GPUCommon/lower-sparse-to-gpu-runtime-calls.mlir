// RUN: mlir-opt %s --gpu-to-llvm='use-opaque-pointers=1' | FileCheck %s

module attributes {gpu.container_module} {

  // CHECK-LABEL: func @matvec
  // CHECK: llvm.call @mgpuStreamCreate
  // CHECK: llvm.call @mgpuMemAlloc
  // CHECK: llvm.call @mgpuMemAlloc
  // CHECK: llvm.call @mgpuCreateSparseEnv
  // CHECK: llvm.call @mgpuCreateCoo
  // CHECK: llvm.call @mgpuCreateDnVec
  // CHECK: llvm.call @mgpuSpMVBufferSize
  // CHECK: llvm.call @mgpuSpM
  // CHECK: llvm.call @mgpuDestroySpMat
  // CHECK: llvm.call @mgpuDestroyDnVec
  // CHECK: llvm.call @mgpuDestroySparseEnv
  // CHECK: llvm.call @mgpuStreamSynchronize
  // CHECK: llvm.call @mgpuStreamDestroy
  func.func @matvec(%arg0: index) {
    %token0 = gpu.wait async
    %mem1, %token1 = gpu.alloc async [%token0] (%arg0) : memref<?xindex>
    %mem2, %token2 = gpu.alloc async [%token1] (%arg0) : memref<?xf64>
    %env, %token3 = gpu.create_sparse_env async [%token2]
    %spmat, %token4 = gpu.create_coo async [%token3] %arg0, %arg0, %arg0, %mem1, %mem1, %mem2 : memref<?xindex>, memref<?xindex>, memref<?xf64>
    %dnvec, %token5 = gpu.create_dn_vec async [%token4] %mem2, %arg0 : memref<?xf64>
    %bufferSz, %token6 = gpu.spmv_buffer_size async [%token5] %env, %spmat, %dnvec, %dnvec
    %token7 = gpu.spmv async [%token6] %env, %spmat, %dnvec, %dnvec, %mem2 : memref<?xf64>
    %token8 = gpu.destroy_sp_mat async [%token7] %spmat
    %token9 = gpu.destroy_dn_vec async [%token8] %dnvec
    %token10 = gpu.destroy_sparse_env async [%token9] %env
    gpu.wait [%token10]
    return
  }

}


