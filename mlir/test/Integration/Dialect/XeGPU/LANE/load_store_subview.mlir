// RUN: mlir-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=lane" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

module @subview attributes {gpu.container_module} {
  gpu.module @kernel {
    gpu.func @subview(%src: memref<256xf32>, %dst: memref<256xf32>) kernel {
      %src_subview = memref.subview %src[5] [251] [1] : memref<256xf32> to memref<251xf32, strided<[1], offset: 5>>
      %dst_subview = memref.subview %dst[10] [246] [1] : memref<256xf32> to memref<246xf32, strided<[1], offset: 10>>
      %lane_id = gpu.lane_id
      %mask = arith.constant 1 : i1
      %loaded = xegpu.load %src_subview[%lane_id], %mask : memref<251xf32, strided<[1], offset: 5>>, index, i1 -> f32
      xegpu.store %loaded, %dst_subview[%lane_id], %mask : f32, memref<246xf32, strided<[1], offset: 10>>, index, i1
      gpu.return
    }
  }
  func.func @test(%src: memref<256xf32>, %dst: memref<256xf32>) -> memref<256xf32> {
    %memref_src = gpu.alloc  () : memref<256xf32>
    gpu.memcpy %memref_src, %src : memref<256xf32>, memref<256xf32>
    %memref_dst = gpu.alloc  () : memref<256xf32>
    gpu.memcpy %memref_dst, %dst : memref<256xf32>, memref<256xf32>
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    gpu.launch_func @kernel::@subview blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1) args(%memref_src : memref<256xf32>, %memref_dst : memref<256xf32>)
    gpu.wait // Wait for the kernel to finish.
    gpu.memcpy %dst, %memref_dst : memref<256xf32>, memref<256xf32>
    gpu.dealloc %memref_src : memref<256xf32>
    gpu.dealloc %memref_dst : memref<256xf32>
    return %dst : memref<256xf32>
  }
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %memref_src = memref.alloc() : memref<256xf32>
    %memref_dst = memref.alloc() : memref<256xf32>
    // Initialize source memref
    scf.for %i = %c0 to %c256 step %c1 {
      %val = arith.index_cast %i : index to i32
      %val_float = arith.sitofp %val : i32 to f32
      memref.store %val_float, %memref_src[%i] : memref<256xf32>
    }
    // Initialize destination memref to zero
    scf.for %i = %c0 to %c256 step %c1 {
      %zero = arith.constant 0.0 : f32
      memref.store %zero, %memref_dst[%i] : memref<256xf32>
    }
    // Call test function
    %gpu_result = call @test(%memref_src, %memref_dst) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %gpu_result_casted = memref.cast %gpu_result : memref<256xf32> to memref<*xf32>
    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    call @printMemrefF32(%gpu_result_casted) : (memref<*xf32>) -> ()
    // Deallocate memrefs
    memref.dealloc %memref_src : memref<256xf32>
    memref.dealloc %memref_dst : memref<256xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}
