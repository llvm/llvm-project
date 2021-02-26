// RUN: mlir-cuda-runner %s \
// RUN:   -gpu-to-cubin="gpu-binary-annotation=nvvm.cubin" \
// RUN:   -gpu-async-region -async-ref-counting \
// RUN:   -gpu-to-llvm="gpu-binary-annotation=nvvm.cubin" \
// RUN:   -async-to-async-runtime -convert-async-to-llvm -convert-std-to-llvm \
// RUN:   --shared-libs=%cuda_wrapper_library_dir/libcuda-runtime-wrappers%shlibext \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_async_runtime%shlibext \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext \
// RUN:   --entry-point-result=void -O0 \
// RUN: | FileCheck %s

func @main() {
  %c0    = constant 0 : index
  %c1    = constant 1 : index
  %count = constant 2 : index

  // initialize h0 on host
  %h0 = alloc(%count) : memref<?xi32>
  %h0_unranked = memref_cast %h0 : memref<?xi32> to memref<*xi32>
  gpu.host_register %h0_unranked : memref<*xi32>

  %v0 = constant 42 : i32
  store %v0, %h0[%c0] : memref<?xi32>
  store %v0, %h0[%c1] : memref<?xi32>

  // copy h0 to b0 on device.
  %t0, %f0 = async.execute () -> !async.value<memref<?xi32>> {
    %b0 = gpu.alloc(%count) : memref<?xi32>
    gpu.memcpy %b0, %h0 : memref<?xi32>, memref<?xi32>
    async.yield %b0 : memref<?xi32>
  }

  // copy h0 to b1 and b2 (fork)
  %t1, %f1 = async.execute [%t0] (
    %f0 as %b0 : !async.value<memref<?xi32>>
  ) -> !async.value<memref<?xi32>> {
    %b1 = gpu.alloc(%count) : memref<?xi32>
    gpu.memcpy %b1, %b0 : memref<?xi32>, memref<?xi32>
    async.yield %b1 : memref<?xi32>
  }
  %t2, %f2 = async.execute [%t0] (
    %f0 as %b0 : !async.value<memref<?xi32>>
  ) -> !async.value<memref<?xi32>> {
    %b2 = gpu.alloc(%count) : memref<?xi32>
    gpu.memcpy %b2, %b0 : memref<?xi32>, memref<?xi32>
    async.yield %b2 : memref<?xi32>
  }

  // h0 = b1 + b2 (join).
  %t3 = async.execute [%t1, %t2] (
    %f1 as %b1 : !async.value<memref<?xi32>>,
    %f2 as %b2 : !async.value<memref<?xi32>>
  ) {
    gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
               threads(%tx, %ty, %tz) in (%block_x = %count, %block_y = %c1, %block_z = %c1) {
      %v1 = load %b1[%tx] : memref<?xi32>
      %v2 = load %b2[%tx] : memref<?xi32>
      %sum = addi %v1, %v2 : i32
      store %sum, %h0[%tx] : memref<?xi32>
      gpu.terminator
    }
    async.yield
  }

  async.await %t3 : !async.token
  // CHECK: [84, 84]
  call @print_memref_i32(%h0_unranked) : (memref<*xi32>) -> ()
  return
}

func private @print_memref_i32(memref<*xi32>)
