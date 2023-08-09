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

  // CHECK-LABEL:     func @spgemm
  // CHECK:      %{{.*}} = gpu.wait async
  // CHECK:           %{{.*}}, %{{.*}} = gpu.alloc async [%{{.*}}] (%{{.*}}) : memref<?xindex>
  // CHECK:           %{{.*}}, %{{.*}} = gpu.alloc async [%{{.*}}] (%{{.*}}) : memref<?xf64>
  // CHECK:           %{{.*}}, %{{.*}} = gpu.create_csr async [%{{.*}}] %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : memref<?xindex>, memref<?xindex>, memref<?xf64>
  // CHECK:           %{{.*}}, %{{.*}} = gpu.create_csr async [%{{.*}}] %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : memref<?xindex>, memref<?xindex>, memref<?xf64>
  // CHECK:           %{{.*}}, %{{.*}} = gpu.create_csr async [%{{.*}}] %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : memref<?xindex>, memref<?xindex>, memref<?xf64>
  // CHECK:           %{{.*}}, %{{.*}} = gpu.spgemm_create_descr async [%{{.*}}]
  // CHECK:           %{{.*}} = memref.alloc() : memref<0xi8>
  // CHECK:           %{{.*}} = arith.constant 0 : index
  // CHECK:           %{{.*}}, %{{.*}} = gpu.spgemm_work_estimation_or_compute async [%{{.*}}]{{{.*}}} %{{.*}}, %{{.*}}, %{{.*}},  ALG2, %{{.*}}, %{{.*}}, %{{.*}} : f32 into memref<0xi8>
  // CHECK:           %{{.*}}, %{{.*}} = gpu.alloc async [%{{.*}}] (%{{.*}}) : memref<?xi8>
  // CHECK:           %{{.*}}, %{{.*}} = gpu.spgemm_work_estimation_or_compute async [%{{.*}}]{{{.*}}} %{{.*}}, %{{.*}}, %{{.*}},  ALG2, %{{.*}}, %{{.*}}, %{{.*}} : f32 into memref<?xi8>
  // CHECK:           %{{.*}}, %{{.*}}, %{{.*}} = gpu.spgemm_estimate_memory async [%{{.*}}] %{{.*}}, %{{.*}}, %{{.*}},  ALG2, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32 into memref<0xi8>
  // CHECK:           %{{.*}}, %{{.*}} = gpu.alloc async [%{{.*}}] (%{{.*}}) : memref<?xi8>
  // CHECK:           %{{.*}}, %{{.*}}, %{{.*}} = gpu.spgemm_estimate_memory async [%{{.*}}] %{{.*}}, %{{.*}}, %{{.*}},  ALG2, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32 into memref<?xi8>
  // CHECK:           %{{.*}}, %{{.*}} = gpu.alloc async [%{{.*}}] (%{{.*}}) : memref<?xi8>
  // CHECK:           %{{.*}}, %{{.*}} = gpu.spgemm_work_estimation_or_compute async [%{{.*}}]{{{.*}}} %{{.*}}, %{{.*}}, %{{.*}},  ALG2, %{{.*}}, %{{.*}}, %{{.*}} : f32 into memref<?xi8>
  // CHECK:           %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} = gpu.spgemm_get_size async [%{{.*}}] %{{.*}}
  // CHECK:           %{{.*}}, %{{.*}} = gpu.alloc async [%{{.*}}] (%{{.*}}) : memref<?xi32>
  // CHECK:           %{{.*}}, %{{.*}} = gpu.alloc async [%{{.*}}] (%{{.*}}) : memref<?xf32>
  // CHECK:           gpu.wait [%{{.*}}]
  // CHECK:           gpu.spgemm_copy  %{{.*}}, %{{.*}}, %{{.*}},  ALG2, %{{.*}} : f32
  // CHECK:           gpu.destroy_sp_mat  %{{.*}}
  // CHECK:           gpu.destroy_sp_mat  %{{.*}}
  // CHECK:           gpu.destroy_sp_mat  %{{.*}}
  // CHECK:           return
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
                            %spmatC, ALG2, %spgemmDesc, %c0, 
                            %alloc: f32 into memref<0xi8>
    %buf1, %token8 = gpu.alloc async [%token7] (%bufferSz1) : memref<?xi8>
    %bufferSz1_1, %token9 = gpu.spgemm_work_estimation_or_compute async 
                              [%token8]{WORK_ESTIMATION} %spmatA, %spmatB, 
                              %spmatC, ALG2, %spgemmDesc, %bufferSz1, 
                              %buf1: f32 into memref<?xi8>
    %bufferSz3, %dummy, %token10 = gpu.spgemm_estimate_memory async [%token9] 
                                     %spmatA, %spmatB, %spmatC, ALG2, 
                                     %spgemmDesc, %c0, %c0, 
                                     %alloc: f32 into memref<0xi8>
    %buf3, %token11 = gpu.alloc async [%token10] (%bufferSz3) : memref<?xi8>
    %bufferSz3_2, %bufferSz2, %token12 = gpu.spgemm_estimate_memory async 
                                          [%token11] %spmatA, %spmatB, %spmatC,
                                          ALG2, %spgemmDesc, %bufferSz3, %c0,
                                          %buf3: f32 into memref<?xi8>
    %buf2, %token13 = gpu.alloc async [%token12] (%bufferSz2) : memref<?xi8>
    %bufferSz2_2, %token14 = gpu.spgemm_work_estimation_or_compute async 
                               [%token13]{COMPUTE} %spmatA, %spmatB, %spmatC, 
                               ALG2, %spgemmDesc, %bufferSz2, 
                               %buf2: f32 into memref<?xi8>
    %rows, %cols, %nnz, %token15 = gpu.spgemm_get_size async [%token14] %spmatC
    %mem_columns, %token16 = gpu.alloc async [%token15] (%cols) : memref<?xi32>
    %mem_values, %token17 = gpu.alloc async [%token16] (%nnz) : memref<?xf32>
    gpu.wait [%token17]
    gpu.spgemm_copy %spmatA, %spmatB, %spmatC, ALG2, %spgemmDesc: f32
    gpu.destroy_sp_mat %spmatA
    gpu.destroy_sp_mat %spmatB
    gpu.destroy_sp_mat %spmatC
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

}


