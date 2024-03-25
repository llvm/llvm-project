// RUN: mlir-opt %s --gpu-to-llvm | FileCheck %s

module attributes {gpu.container_module} {

  // CHECK-LABEL: func @matmul
  // CHECK: llvm.call @mgpuStreamCreate
  // CHECK: llvm.call @mgpuMemAlloc
  // CHECK: llvm.call @mgpuMemAlloc
  // CHECK: llvm.call @mgpuCusparseLtCreate2To4SpMat
  // CHECK: llvm.call @mgpuCreateCuSparseLtDnMat
  // CHECK: llvm.call @mgpuCuSparseLtSpMMBufferSize
  // CHECK: llvm.call @mgpuCuSparseLtSpMM
  // CHECK: llvm.call @mgpuDestroyCuSparseLtSpMat
  // CHECK: llvm.call @mgpuDestroyCuSparseLtDnMat
  // CHECK: llvm.call @mgpuStreamSynchronize
  // CHECK: llvm.call @mgpuStreamDestroy
  func.func @matmul(%arg0: index) {
    %token0 = gpu.wait async
    %mem1, %token1 = gpu.alloc async [%token0] (%arg0) : memref<?xf16>
    %mem2, %token2 = gpu.alloc async [%token1] (%arg0) : memref<?xf16>
    %spmat, %token4 = gpu.create_2to4_spmat async [%token2] {PRUNE_AND_CHECK} %arg0, %arg0, %mem1:  memref<?xf16>
    %dnmat, %token5 = gpu.create_dn_tensor async [%token4]  %mem2, %arg0, %arg0 : index, index into memref<?xf16>
    %bufferSz0, %bufferSz1, %bufferSz2, %token6 = gpu.spmm_buffer_size async [%token5] %spmat, %dnmat, %dnmat : index,index,index into f16
    %token7 = gpu.spmm async [%token6] %spmat, %dnmat, %dnmat, %mem2, %mem2, %mem2 : memref<?xf16>,memref<?xf16>,memref<?xf16> into f16
    %token8 = gpu.destroy_sp_mat async [%token7] %spmat
    %token9 = gpu.destroy_dn_tensor async [%token8] %dnmat
    gpu.wait [%token9]
    return
  }

}
