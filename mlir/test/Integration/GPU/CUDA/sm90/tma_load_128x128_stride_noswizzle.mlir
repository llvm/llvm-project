// RUN: mlir-opt %s \
// RUN:  -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_90 cubin-features=+ptx80 opt-level=3" \
// RUN:  | mlir-runner \
// RUN:   --shared-libs=%mlir_cuda_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --entry-point-result=void \
// RUN:  | FileCheck %s

// CHECK: Correct Results :8192
// CHECK: Incorrect Results :0

module {
  func.func @main() {
    %c10000000 = arith.constant 10000000 : index
    %false = arith.constant false
    %c32768 = arith.constant 32768 : index
    %c31_i32 = arith.constant 31 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c5_i32 = arith.constant 5 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %c2 = arith.constant 2 : index
    %c32768_i32 = arith.constant 32768 : i32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.mlir.constant(128 : i64) : i64
    %2 = llvm.mlir.constant(0 : i64) : i64
    %f0 = arith.constant 0.0 : f16
    %f123 = arith.constant 1.123 : f16
    
    %srcMemref_host = memref.alloc() : memref<128x128xf16>
    %dstMemref_host = memref.alloc() : memref<128x128xf16>
    scf.for %arg0 = %c0 to %c128 step %c1 {
      scf.for %arg1 = %c0 to %c64 step %c1 {
        %d1 = arith.index_cast %arg0 : index to i32
        %d2 = arith.index_cast %arg1 : index to i32
        %d3 = arith.sitofp %d1 : i32 to f16
        %d4 = arith.sitofp %d2 : i32 to f16
        %d5 = arith.addf %d3, %f123 : f16
        %d6 = arith.constant 3.12 : f16
        %d7 = arith.mulf %d5, %d6 : f16
        %d8 = arith.addf %d7, %d5 : f16
        %d9 = arith.constant 0.178 : f16
        %d10 = arith.divf %d9, %d8 : f16
        memref.store %d10, %srcMemref_host[%arg0, %arg1] : memref<128x128xf16>
        memref.store %f0, %dstMemref_host[%arg0, %arg1] : memref<128x128xf16>
      }
    }

    %s1 = gpu.wait async
    %srcMemref, %s2 = gpu.alloc async [%s1] () : memref<128x128xf16>
    %dstMemref, %s3 = gpu.alloc async [%s2] () : memref<128x128xf16>
    %s4 = gpu.memcpy async [%s3] %srcMemref, %srcMemref_host : memref<128x128xf16>, memref<128x128xf16>
    %s5 = gpu.memcpy async [%s4] %dstMemref, %dstMemref_host : memref<128x128xf16>, memref<128x128xf16>

    %expand_shape = memref.expand_shape %srcMemref [[0, 1], [2, 3]] output_shape [2, 64, 2, 64] : memref<128x128xf16> into memref<2x64x2x64xf16>
    %transpose = memref.transpose %expand_shape (d0, d1, d2, d3) -> (d0, d2, d1, d3) : memref<2x64x2x64xf16> to memref<2x2x64x64xf16, strided<[8192, 64, 128, 1]>>
    %cast = memref.cast %transpose : memref<2x2x64x64xf16, strided<[8192, 64, 128, 1]>> to memref<*xf16>
    %24 = nvgpu.tma.create.descriptor %cast box[%c2, %c2, %c64, %c64] : memref<*xf16> -> <tensor = memref<2x2x64x64xf16, 3>, swizzle = none, l2promo = none, oob = zero, interleave = none>
    
    gpu.launch 
      blocks(%arg2, %arg3, %arg4) in (%arg8 = %c1, %arg9 = %c1, %arg10 = %c1) 
      threads(%arg5, %arg6, %arg7) in (%arg11 = %c128, %arg12 = %c1, %arg13 = %c1) 
      dynamic_shared_memory_size %c32768_i32 
    {
      %26 = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
      %view = memref.view %26[%c0][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<2x2x64x64xf16, #gpu.address_space<workgroup>>
      %27 = nvgpu.mbarrier.create -> <memorySpace = #gpu.address_space<workgroup>>
      %thread_id_x = gpu.thread_id  x
      %28 = arith.index_cast %thread_id_x : index to i32
      %29 = arith.shrui %28, %c5_i32 : i32
      %30 = nvvm.shfl.sync  idx %c-1_i32, %29, %c0_i32, %c31_i32 : i32 -> i32
      %31 = arith.cmpi eq, %30, %c0_i32 : i32
      %32 = nvvm.elect.sync -> i1
      %33 = arith.andi %31, %32 : i1
      scf.if %33 {
        nvgpu.mbarrier.init %27[%c0], %c1 : <memorySpace = #gpu.address_space<workgroup>>
      }
      %34 = nvvm.shfl.sync  idx %c-1_i32, %29, %c0_i32, %c31_i32 : i32 -> i32
      %35 = arith.cmpi eq, %34, %c0_i32 : i32
      %36 = nvvm.elect.sync -> i1
      %37 = arith.andi %35, %36 : i1
      scf.if %37 {
        nvgpu.mbarrier.arrive.expect_tx %27[%c0], %c32768 : <memorySpace = #gpu.address_space<workgroup>>
        nvgpu.tma.async.load %24[%c0, %c0, %c0, %c0], %27[%c0] to %view : <tensor = memref<2x2x64x64xf16, 3>, swizzle = none, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>> -> memref<2x2x64x64xf16, #gpu.address_space<workgroup>>
      }
      nvgpu.mbarrier.try_wait.parity %27[%c0], %false, %c10000000 : <memorySpace = #gpu.address_space<workgroup>>
      scf.for %arg14 = %c0 to %c2 step %c1 {
        scf.for %arg15 = %c0 to %c2 step %c1 {
          %38 = arith.muli %arg14, %c64 : index
          %39 = arith.muli %arg15, %c64 : index
          %subview = memref.subview %view[%arg14, %arg15, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : memref<2x2x64x64xf16, #gpu.address_space<workgroup>> to memref<64x64xf16, strided<[64, 1], offset: ?>, #gpu.address_space<workgroup>>
          %subview_0 = memref.subview %dstMemref[%38, %39] [64, 64] [1, 1] : memref<128x128xf16> to memref<64x64xf16, strided<[128, 1], offset: ?>>
          %block_dim_x = gpu.block_dim  x
          %thread_id_y = gpu.thread_id  y
          %40 = arith.muli %thread_id_y, %block_dim_x : index
          %41 = arith.addi %thread_id_x, %40 : index
          %block_dim_y = gpu.block_dim  y
          %42 = arith.muli %block_dim_x, %block_dim_y : index
          %thread_id_z = gpu.thread_id  z
          %43 = arith.muli %thread_id_z, %42 : index
          %44 = arith.addi %41, %43 : index
          %45 = arith.cmpi eq, %44, %c0 : index
          scf.if %45 {
            scf.for %arg16 = %c0 to %c64 step %c1 {
              scf.for %arg17 = %c0 to %c64 step %c1 {
                %46 = memref.load %subview[%arg16, %arg17] : memref<64x64xf16, strided<[64, 1], offset: ?>, #gpu.address_space<workgroup>>
                memref.store %46, %subview_0[%arg16, %arg17] : memref<64x64xf16, strided<[128, 1], offset: ?>>
              }
            }
          }
          gpu.barrier
        }
      }
      gpu.terminator
    }

    %s6 = gpu.memcpy async [%s5] %dstMemref_host, %dstMemref  : memref<128x128xf16>, memref<128x128xf16>
    gpu.wait [%s6]

   %errorCount, %correctCount =  scf.for %arg0 = %c0 to %c128 step %c1 iter_args(%ec1 = %c0, %cc1 = %c0) -> (index,index) {
      %ec2, %cc2 = scf.for %arg1 = %c0 to %c64 step %c1 iter_args(%ec2 = %ec1, %cc2 = %cc1) -> (index, index) { 
        %v1 = memref.load %dstMemref_host[%arg0, %arg1] : memref<128x128xf16>
        %v2 = memref.load %srcMemref_host[%arg0, %arg1] : memref<128x128xf16>
        %p = arith.cmpf one, %v1, %v2 : f16        
        %ec3, %cc3 = scf.if %p -> (index, index) {
          %ec3 = arith.addi %ec2, %c1 : index
          scf.yield %ec3, %cc2 : index, index
        } else {
          %cc3 = arith.addi %cc2, %c1 : index
          scf.yield %ec2, %cc3 : index, index
        }
      scf.yield %ec3, %cc3 : index,index
      }
      scf.yield %ec2, %cc2 : index,index
    }
    
    vector.print str "Correct Results :"
    vector.print %correctCount : index
    vector.print str "Incorrect Results :"
    vector.print %errorCount : index
    return
  }
}
