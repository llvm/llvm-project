// RUN: mlir-opt %s -split-input-file | mlir-opt -split-input-file | FileCheck %s

module attributes {gpu.container_module} {

  // CHECK-LABEL: func @matvec
  // CHECK: %{{.*}} = gpu.wait async
  // CHECK: %{{.*}}, %{{.*}} = gpu.alloc async [%{{.*}}] (%{{.*}}) : memref<?xindex>
  // CHECK: %{{.*}}, %{{.*}} = gpu.alloc async [%{{.*}}] (%{{.*}}) : memref<?xf64>
  // CHECK: %{{.*}}, %{{.*}} = gpu.create_coo async [%{{.*}}] %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : memref<?xindex>, memref<?xindex>, memref<?xf64>
  // CHECK: %{{.*}}, %{{.*}} = gpu.create_dn_tensor async [%{{.*}}] %{{.*}}, %{{.*}} : index into memref<?xf64>
  // CHECK: %{{.*}}, %{{.*}} = gpu.spmv_buffer_size async [%{{.*}}] %{{.*}}, %{{.*}}, %{{.*}} into f64
  // CHECK: %{{.*}} = gpu.spmv async [%{{.*}}] %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : memref<?xf64> into f64
  // CHECK: %{{.*}} = gpu.destroy_sp_mat async [%{{.*}}] %{{.*}}
  // CHECK: %{{.*}} = gpu.destroy_dn_tensor async [%{{.*}}] %{{.*}}
  // CHECK: gpu.wait [%{{.*}}]
  // CHECK: return
  func.func @matvec(%arg0: index) {
    %token0 = gpu.wait async
    %mem1, %token1 = gpu.alloc async [%token0] (%arg0) : memref<?xindex>
    %mem2, %token2 = gpu.alloc async [%token1] (%arg0) : memref<?xf64>
    %spmat, %token4 = gpu.create_coo async [%token2] %arg0, %arg0, %arg0, %mem1, %mem1, %mem2 : memref<?xindex>, memref<?xindex>, memref<?xf64>
    %dnvec, %token5 = gpu.create_dn_tensor async [%token4] %mem2, %arg0 : index into memref<?xf64>
    %bufferSz, %token6 = gpu.spmv_buffer_size async [%token5] %spmat, %dnvec, %dnvec into f64
    %token7 = gpu.spmv async [%token6] %spmat, %dnvec, %dnvec, %mem2 : memref<?xf64> into f64
    %token8 = gpu.destroy_sp_mat async [%token7] %spmat
    %token9 = gpu.destroy_dn_tensor async [%token8] %dnvec
    gpu.wait [%token9]
    return
  }

  // CHECK-LABEL: func @matmul
  // CHECK: %{{.*}} = gpu.wait async
  // CHECK: %{{.*}}, %{{.*}} = gpu.alloc async [%{{.*}}] (%{{.*}}) : memref<?xindex>
  // CHECK: %{{.*}}, %{{.*}} = gpu.alloc async [%{{.*}}] (%{{.*}}) : memref<?xf64>
  // CHECK: %{{.*}}, %{{.*}} = gpu.create_csr async [%{{.*}}] %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : memref<?xindex>, memref<?xindex>, memref<?xf64>
  // CHECK: %{{.*}}, %{{.*}} = gpu.create_dn_tensor async [%{{.*}}] %{{.*}}, %{{.*}}, %{{.*}} : index, index into memref<?xf64>
  // CHECK: %{{.*}}, %{{.*}} = gpu.spmm_buffer_size async [%{{.*}}] %{{.*}}, %{{.*}}, %{{.*}} into f64
  // CHECK: %{{.*}} = gpu.spmm async [%{{.*}}] %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : memref<?xf64> into f64
  // CHECK: %{{.*}} = gpu.destroy_sp_mat async [%{{.*}}] %{{.*}}
  // CHECK: %{{.*}} = gpu.destroy_dn_tensor async [%{{.*}}] %{{.*}}
  // CHECK: gpu.wait [%{{.*}}]
  // CHECK: return
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

  // CHECK-LABEL: func @spgemm
  // CHECK: %{{.*}} = gpu.wait async
  // CHECK: %{{.*}}, %{{.*}} = gpu.alloc async [%{{.*}}] (%{{.*}}) : memref<?xindex>
  // CHECK: %{{.*}}, %{{.*}} = gpu.alloc async [%{{.*}}] (%{{.*}}) : memref<?xf32>
  // CHECK: %{{.*}}, %{{.*}} = gpu.create_csr async [%{{.*}}] %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : memref<?xindex>, memref<?xindex>, memref<?xf32>
  // CHECK: %{{.*}}, %{{.*}} = gpu.create_csr async [%{{.*}}] %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : memref<?xindex>, memref<?xindex>, memref<?xf32>
  // CHECK: %{{.*}}, %{{.*}} = gpu.create_csr async [%{{.*}}] %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : memref<?xindex>, memref<?xindex>, memref<?xf32>
  // CHECK: %{{.*}}, %{{.*}} = gpu.spgemm_create_descr async [%{{.*}}]
  // CHECK: %{{.*}}, %{{.*}} = gpu.spgemm_work_estimation_or_compute async [%{{.*}}]{ WORK_ESTIMATION} %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32 into memref<0xi8>
  // CHECK: %{{.*}}, %{{.*}} = gpu.spgemm_work_estimation_or_compute async [%{{.*}}]{ COMPUTE} %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32 into memref<0xi8>
  // CHECK: %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} = gpu.spmat_get_size async [%{{.*}}] %{{.*}}
  // CHECK: %{{.*}} = gpu.set_csr_pointers async [%{{.*}}] %{{.*}}, {{.*}}, {{.*}}, {{.*}} : memref<?xindex>, memref<?xindex>, memref<?xf32>
  // CHECK: %{{.*}} = gpu.spgemm_copy async [%{{.*}}] %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
  // CHECK: %{{.*}} = gpu.spgemm_destroy_descr async [%{{.*}}] %{{.*}}
  // CHECK: %{{.*}} = gpu.destroy_sp_mat async [%{{.*}}] %{{.*}}
  // CHECK: %{{.*}} = gpu.destroy_sp_mat async [%{{.*}}] %{{.*}}
  // CHECK: %{{.*}} = gpu.destroy_sp_mat async [%{{.*}}] %{{.*}}
  // CHECK: gpu.wait [%{{.*}}]
  // CHECK: return
  func.func @spgemm(%arg0: index) {
    %token0 = gpu.wait async
    %mem1, %token1 = gpu.alloc async [%token0] (%arg0) : memref<?xindex>
    %mem2, %token2 = gpu.alloc async [%token1] (%arg0) : memref<?xf32>
    %spmatA, %token3 = gpu.create_csr async [%token2] %arg0, %arg0, %arg0, %mem1, %mem1, %mem2 : memref<?xindex>, memref<?xindex>, memref<?xf32>
    %spmatB, %token4 = gpu.create_csr async [%token3] %arg0, %arg0, %arg0, %mem1, %mem1, %mem2 : memref<?xindex>, memref<?xindex>, memref<?xf32>
    %spmatC, %token5 = gpu.create_csr async [%token4] %arg0, %arg0, %arg0, %mem1, %mem1, %mem2 : memref<?xindex>, memref<?xindex>, memref<?xf32>
    %spgemmDesc, %token6 = gpu.spgemm_create_descr async [%token5]
    %alloc = memref.alloc() : memref<0xi8>  // nullptr
    %c0 = arith.constant 0 : index
    %bufferSz1, %token7 = gpu.spgemm_work_estimation_or_compute async
                            [%token6]{WORK_ESTIMATION}
                            %spmatA, %spmatB, %spmatC,
			    %spgemmDesc, %c0, %alloc: f32 into memref<0xi8>
    %bufferSz2, %token8 = gpu.spgemm_work_estimation_or_compute async
                               [%token7]{COMPUTE}
			       %spmatA, %spmatB, %spmatC,
                               %spgemmDesc, %c0, %alloc: f32 into memref<0xi8>
    %rows, %cols, %nnz, %token9 = gpu.spmat_get_size async [%token8] %spmatC
    %token10 = gpu.set_csr_pointers async [%token8] %spmatC, %mem1, %mem1, %mem2 : memref<?xindex>, memref<?xindex>, memref<?xf32>
    %token11 = gpu.spgemm_copy async [%token10] %spmatA, %spmatB, %spmatC, %spgemmDesc: f32
    %token12 = gpu.spgemm_destroy_descr async [%token11] %spgemmDesc
    %token13 = gpu.destroy_sp_mat async [%token12] %spmatA
    %token14 = gpu.destroy_sp_mat async [%token13] %spmatB
    %token15 = gpu.destroy_sp_mat async [%token14] %spmatC
    gpu.wait [%token15]
    return
  }

  // CHECK-LABEL: func @sddmm
  // CHECK: %{{.*}} = gpu.wait async
  // CHECK: %{{.*}}, %{{.*}} = gpu.alloc async [%{{.*}}] (%{{.*}}) : memref<?xindex>
  // CHECK: %{{.*}}, %{{.*}} = gpu.alloc async [%{{.*}}] (%{{.*}}) : memref<?xf64>
  // CHECK: %{{.*}}, %{{.*}} = gpu.create_csr async [%{{.*}}] %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : memref<?xindex>, memref<?xindex>, memref<?xf64>
  // CHECK: %{{.*}}, %{{.*}} = gpu.create_dn_tensor async [%{{.*}}] %{{.*}}, %{{.*}}, %{{.*}} : index, index into memref<?xf64>
  // CHECK: %{{.*}}, %{{.*}} = gpu.sddmm_buffer_size async [%{{.*}}] %{{.*}}, %{{.*}}, %{{.*}}  into f64
  // CHECK: %{{.*}} = gpu.sddmm async [%{{.*}}] %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : memref<?xf64>  into f64
  // CHECK: %{{.*}} = gpu.destroy_sp_mat async [%{{.*}}] %{{.*}}
  // CHECK: %{{.*}} = gpu.destroy_dn_tensor async [%{{.*}}] %{{.*}}
  // CHECK: gpu.wait [%{{.*}}]
  // CHECK: return
  func.func @sddmm(%arg0: index) {
    %token0 = gpu.wait async
    %mem1, %token1 = gpu.alloc async [%token0] (%arg0) : memref<?xindex>
    %mem2, %token2 = gpu.alloc async [%token1] (%arg0) : memref<?xf64>
    %spmat, %token4 = gpu.create_csr async [%token2] %arg0, %arg0, %arg0, %mem1, %mem1, %mem2 : memref<?xindex>, memref<?xindex>, memref<?xf64>
    %dnmat, %token5 = gpu.create_dn_tensor async [%token4] %mem2, %arg0, %arg0 : index, index into memref<?xf64>
    %bufferSz, %token6 = gpu.sddmm_buffer_size async [%token5] %dnmat, %dnmat, %spmat into f64
    %token7 = gpu.sddmm async [%token6] %dnmat, %dnmat, %spmat, %mem2 : memref<?xf64> into f64
    %token8 = gpu.destroy_sp_mat async [%token7] %spmat
    %token9 = gpu.destroy_dn_tensor async [%token8] %dnmat
    gpu.wait [%token9]
    return
  }

  // CHECK-LABEL: func @csc_and_bsr
  // CHECK: %{{.*}} = gpu.wait async
  // CHECK: %{{.*}}, %{{.*}} = gpu.alloc async [%{{.*}}] (%{{.*}}) : memref<?xindex>
  // CHECK: %{{.*}}, %{{.*}} = gpu.alloc async [%{{.*}}] (%{{.*}}) : memref<?xf64>
  // CHECK: %{{.*}}, %{{.*}} = gpu.create_csc async [%{{.*}}] %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : memref<?xindex>, memref<?xindex>, memref<?xf64>
  // CHECK: %{{.*}}, %{{.*}} = gpu.create_bsr async [%{{.*}}] %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : memref<?xindex>, memref<?xindex>, memref<?xf64>
  // CHECK: gpu.wait [%{{.*}}]
  // CHECK: return
  func.func @csc_and_bsr(%arg0: index) {
    %token0 = gpu.wait async
    %mem1, %token1 = gpu.alloc async [%token0] (%arg0) : memref<?xindex>
    %mem2, %token2 = gpu.alloc async [%token1] (%arg0) : memref<?xf64>
    %csc, %token3 = gpu.create_csc async [%token2]
      %arg0, %arg0, %arg0, %mem1, %mem1, %mem2
      : memref<?xindex>, memref<?xindex>, memref<?xf64>
    %bsr, %token4 = gpu.create_bsr async [%token3]
      %arg0, %arg0, %arg0, %arg0, %arg0, %mem1, %mem1, %mem2
      : memref<?xindex>, memref<?xindex>, memref<?xf64>
    %token5 = gpu.destroy_sp_mat async [%token4] %csc
    %token6 = gpu.destroy_sp_mat async [%token5] %bsr
    gpu.wait [%token6]
    return
  }

}
