// RUN: mlir-opt %s \
// RUN:  -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_90a cubin-features=+ptx80 opt-level=3" \
// RUN:  | mlir-cpu-runner \
// RUN:   --shared-libs=%mlir_cuda_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --entry-point-result=void \
// RUN:  | FileCheck %s

// CHECK: clusterIdx: (1, 1, 0) in Cluster Dimension: (2, 2, 1) blockIdx: (3, 3, 0) 

module attributes {gpu.container_module} {
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index    
    gpu.launch_func  @gpumodule::@kernel_cluster clusters in(%c2,%c2,%c1)  blocks in (%c4, %c4, %c1) threads in (%c1, %c1, %c1)  
    return
  }
  gpu.module @gpumodule {
    gpu.func @kernel_cluster() kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 2, 2, 1>} {
      %cidX = gpu.cluster_id  x
      %cidY = gpu.cluster_id  y
      %cidZ = gpu.cluster_id  z
      %cdimX = gpu.cluster_dim  x
      %cdimY = gpu.cluster_dim  y
      %cdimZ = gpu.cluster_dim  z
      %bidX = gpu.block_id  x
      %bidY = gpu.block_id  y
      %bidZ = gpu.block_id  z
      %cidX_i32 = index.casts %cidX : index to i32
      %cidY_i32 = index.casts %cidY : index to i32
      %cidZ_i32 = index.casts %cidZ : index to i32
      %cdimX_i32 = index.casts %cdimX : index to i32
      %cdimY_i32 = index.casts %cdimY : index to i32
      %cdimZ_i32 = index.casts %cdimZ : index to i32
      %bidX_i32 = index.casts %bidX : index to i32
      %bidY_i32 = index.casts %bidY : index to i32
      %bidZ_i32 = index.casts %bidZ : index to i32

      %c3 = arith.constant 3 : index
      %cnd1 =  arith.cmpi eq, %bidX, %c3 : index
      %cnd2 =  arith.cmpi eq, %bidY, %c3 : index
      scf.if %cnd1 {
        scf.if %cnd2 {
          gpu.printf "clusterIdx: (%d, %d, %d) in Cluster Dimension: (%d, %d, %d) blockIdx: (%d, %d, %d) \n" 
            %cidX_i32,
            %cidY_i32,
            %cidZ_i32,
            %cdimX_i32,
            %cdimY_i32,
            %cdimZ_i32,
            %bidX_i32,
            %bidY_i32,
            %bidZ_i32      
            : 
            i32, i32, i32, i32, i32, i32, i32, i32, i32
        }
      }
      
      gpu.return
    }
  }
}

