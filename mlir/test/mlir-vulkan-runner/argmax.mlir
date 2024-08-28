// RUN: mlir-vulkan-runner %s \
// RUN:  --shared-libs=%vulkan-runtime-wrappers,%mlir_runner_utils \
// RUN:  --entry-point-result=void | FileCheck %s

// This kernel computes the argmax (index of the maximum element) from an array
// of integers. Each thread computes a lane maximum using a single `scf.for`.
// Then `gpu.subgroup_reduce` is used to find the maximum across the entire
// subgroup, which is then used by SPIR-V subgroup ops to compute the argmax
// of the entire input array. Note that this kernel only works if we have a
// single workgroup.

// CHECK: [15]
module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.3, [Shader, Groups, GroupNonUniformArithmetic, GroupNonUniformBallot], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {
  gpu.module @kernels {
    gpu.func @kernel_argmax(%input : memref<128xi32>, %output : memref<1xi32>, %total_count_buf : memref<1xi32>) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 1, 1]>} {
      %idx0 = arith.constant 0 : index
      %idx1 = arith.constant 1 : index

      %total_count = memref.load %total_count_buf[%idx0] : memref<1xi32>
      %lane_count_idx = gpu.subgroup_size : index
      %lane_count_i32 = index.castu %lane_count_idx : index to i32
      %lane_id_idx = gpu.thread_id x
      %lane_id_i32 = index.castu %lane_id_idx : index to i32
      %lane_res_init = arith.constant 0 : i32
      %lane_max_init = memref.load %input[%lane_id_idx] : memref<128xi32>
      %num_batches_i32 = arith.divui %total_count, %lane_count_i32 : i32
      %num_batches_idx = index.castu %num_batches_i32 : i32 to index

      %lane_res, %lane_max = scf.for %iter = %idx1 to %num_batches_idx step %idx1
      iter_args(%lane_res_iter = %lane_res_init, %lane_max_iter = %lane_max_init) -> (i32, i32) {
        %iter_i32 = index.castu %iter : index to i32
        %mul = arith.muli %lane_count_i32, %iter_i32 : i32
        %idx_i32 = arith.addi %mul, %lane_id_i32 : i32
        %idx = index.castu %idx_i32 : i32 to index
        %elem = memref.load %input[%idx] : memref<128xi32>
        %gt = arith.cmpi sgt, %elem, %lane_max_iter : i32
        %lane_res_next = arith.select %gt, %idx_i32, %lane_res_iter : i32
        %lane_max_next = arith.select %gt, %elem, %lane_max_iter : i32
        scf.yield %lane_res_next, %lane_max_next : i32, i32
      }

      %subgroup_max = gpu.subgroup_reduce maxsi %lane_max : (i32) -> (i32)
      %eq = arith.cmpi eq, %lane_max, %subgroup_max : i32
      %ballot = spirv.GroupNonUniformBallot <Subgroup> %eq : vector<4xi32>
      %lsb = spirv.GroupNonUniformBallotFindLSB <Subgroup> %ballot : vector<4xi32>, i32
      %cond = arith.cmpi eq, %lsb, %lane_id_i32 : i32

      scf.if %cond {
        memref.store %lane_res, %output[%idx0] : memref<1xi32>
      }

      gpu.return
    }
  }

  func.func @main() {
    // Allocate 3 buffers.
    %in_buf = memref.alloc() : memref<128xi32>
    %out_buf = memref.alloc() : memref<1xi32>
    %total_count_buf = memref.alloc() : memref<1xi32>

    // Constants.
    %cst0 = arith.constant 0 : i32
    %idx0 = arith.constant 0 : index
    %idx1 = arith.constant 1 : index
    %idx16 = arith.constant 16 : index
    %idx32 = arith.constant 32 : index
    %idx48 = arith.constant 48 : index
    %idx64 = arith.constant 64 : index
    %idx80 = arith.constant 80 : index
    %idx96 = arith.constant 96 : index
    %idx112 = arith.constant 112 : index

    // Initialize input buffer.
    %in_vec = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xi32>
    vector.store %in_vec, %in_buf[%idx0] : memref<128xi32>, vector<16xi32>
    vector.store %in_vec, %in_buf[%idx16] : memref<128xi32>, vector<16xi32>
    vector.store %in_vec, %in_buf[%idx32] : memref<128xi32>, vector<16xi32>
    vector.store %in_vec, %in_buf[%idx48] : memref<128xi32>, vector<16xi32>
    vector.store %in_vec, %in_buf[%idx64] : memref<128xi32>, vector<16xi32>
    vector.store %in_vec, %in_buf[%idx80] : memref<128xi32>, vector<16xi32>
    vector.store %in_vec, %in_buf[%idx96] : memref<128xi32>, vector<16xi32>
    vector.store %in_vec, %in_buf[%idx112] : memref<128xi32>, vector<16xi32>

    // Initialize output buffer.
    %out_buf2 = memref.cast %out_buf : memref<1xi32> to memref<?xi32>
    call @fillResource1DInt(%out_buf2, %cst0) : (memref<?xi32>, i32) -> ()

    // Total number of scalars.
    %total_count = arith.constant 128 : i32
    %total_count_buf2 = memref.cast %total_count_buf : memref<1xi32> to memref<?xi32>
    call @fillResource1DInt(%total_count_buf2, %total_count) : (memref<?xi32>, i32) -> ()

    // Launch kernel function and print output.
    gpu.launch_func @kernels::@kernel_argmax
        blocks in (%idx1, %idx1, %idx1) threads in (%idx32, %idx1, %idx1)
        args(%in_buf : memref<128xi32>, %out_buf : memref<1xi32>, %total_count_buf : memref<1xi32>)
    %out_buf3 = memref.cast %out_buf2 : memref<?xi32> to memref<*xi32>
    call @printMemrefI32(%out_buf3) : (memref<*xi32>) -> ()
    return
  }
  func.func private @fillResource1DInt(%0 : memref<?xi32>, %1 : i32)
  func.func private @printMemrefI32(%ptr : memref<*xi32>)
}
