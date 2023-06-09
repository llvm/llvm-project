// RUN: mlir-opt %s --gpu-to-llvm='use-opaque-pointers=1' | FileCheck %s

module attributes {gpu.container_module} {

  // CHECK-LABEL: func @matmul
  // CHECK: llvm.call @mgpuStreamCreate
  // CHECK: llvm.call @mgpuMemAlloc
  // CHECK: llvm.call @mgpuMemAlloc
  // CHECK: llvm.call @mgpuCreateSparseLtEnv
  // CHECK: llvm.call @mgpuCusparseLtCreate2To4SpMat
  // CHECK: llvm.call @mgpuCreateCuSparseLtDnMat
  // CHECK: llvm.call @mgpuCuSparseLtSpMMBufferSize
  // CHECK: llvm.call @mgpuCuSparseLtSpMM
  // CHECK: llvm.call @mgpuDestroyCuSparseLtSpMat
  // CHECK: llvm.call @mgpuDestroyCuSparseLtDnMat
  // CHECK: llvm.call @mgpuDestroySparseLtEnv
  // CHECK: llvm.call @mgpuStreamSynchronize
  // CHECK: llvm.call @mgpuStreamDestroy
  func.func @matmul(%arg0: index) {
    %token0 = gpu.wait async
    %mem1, %token1 = gpu.alloc async [%token0] (%arg0) : memref<?xf16>
    %mem2, %token2 = gpu.alloc async [%token1] (%arg0) : memref<?xf16>
    %env, %token3 = gpu.create_sparse_env async [%token2]
    %spmat, %token4 = gpu.create_2to4_spmat async [%token3] %env, %arg0, %arg0, %mem1:  memref<?xf16>
    %dnmat, %token5 = gpu.create_dn_mat async [%token4] %env, %arg0, %arg0, %mem2 : memref<?xf16>
    %bufferSzs, %token6 = gpu.spmm_buffer_size async [%token5] %env, %spmat, %dnmat, %dnmat : tuple<index,index,index> into f16
    %token7 = gpu.spmm async [%token6] %env, %spmat, %dnmat, %dnmat, %mem2, %mem2, %mem2 : memref<?xf16>,memref<?xf16>,memref<?xf16> into f16
    %token8 = gpu.destroy_sp_mat async [%token7] %spmat
    %token9 = gpu.destroy_dn_mat async [%token8] %dnmat
    %token10 = gpu.destroy_sparse_env async [%token9] %env
    gpu.wait [%token10]
    return
  }

}
