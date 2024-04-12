module {
  func.func @gemm_warp_specialized(%arg0: memref<512x1024xf16>, %arg1: memref<1024x256xf16>, %arg2: memref<512x256xf32>) attributes {llvm.emit_c_interface} {
    %0 = gpu.wait async
    %memref, %asyncToken = gpu.alloc async [%0] () : memref<512x1024xf16>
    %memref_0, %asyncToken_1 = gpu.alloc async [%asyncToken] () : memref<1024x256xf16>
    %memref_2, %asyncToken_3 = gpu.alloc async [%asyncToken_1] () : memref<512x256xf32>
    %1 = gpu.memcpy async [%asyncToken_3] %memref, %arg0 : memref<512x1024xf16>, memref<512x1024xf16>
    %2 = gpu.memcpy async [%1] %memref_0, %arg1 : memref<1024x256xf16>, memref<1024x256xf16>
    %3 = gpu.wait async [%2]
    %cast = memref.cast %memref : memref<512x1024xf16> to memref<*xf16>
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %4 = nvgpu.tma.create.descriptor %cast box[%c128, %c64] : memref<*xf16> -> <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
    %cast_4 = memref.cast %memref_0 : memref<1024x256xf16> to memref<*xf16>
    %c64_5 = arith.constant 64 : index
    %c64_6 = arith.constant 64 : index
    %5 = nvgpu.tma.create.descriptor %cast_4 box[%c64_5, %c64_6] : memref<*xf16> -> <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c1_7 = arith.constant 1 : index
    %c1_8 = arith.constant 1 : index
    %c229376_i32 = arith.constant 229376 : i32
    gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c4, %arg10 = %c2, %arg11 = %c1) threads(%arg6, %arg7, %arg8) in (%arg12 = %c256, %arg13 = %c1_7, %arg14 = %c1_8) dynamic_shared_memory_size %c229376_i32 {
      %thread_id_x = gpu.thread_id  x
      %c128_9 = arith.constant 128 : index
      %7 = arith.remui %thread_id_x, %c128_9 : index
      %c0 = arith.constant 0 : index
      %8 = arith.cmpi eq, %7, %c0 : index
      %c128_10 = arith.constant 128 : index
      %9 = arith.divui %thread_id_x, %c128_10 : index
      %c1_11 = arith.constant 1 : index
      %10 = arith.cmpi eq, %9, %c1_11 : index
      %thread_id_x_12 = gpu.thread_id  x
      %c128_13 = arith.constant 128 : index
      %11 = arith.remui %thread_id_x_12, %c128_13 : index
      %c0_14 = arith.constant 0 : index
      %12 = arith.cmpi eq, %11, %c0_14 : index
      %c128_15 = arith.constant 128 : index
      %13 = arith.divui %thread_id_x_12, %c128_15 : index
      %c0_16 = arith.constant 0 : index
      %14 = arith.cmpi eq, %13, %c0_16 : index
      %thread_id_x_17 = gpu.thread_id  x
      %15 = nvgpu.mbarrier.create -> <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7>
      %16 = nvgpu.mbarrier.create -> <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7>
      %c0_18 = arith.constant 0 : index
      %17 = arith.cmpi eq, %thread_id_x_17, %c0_18 : index
      scf.if %17 {
        %c0_19 = arith.constant 0 : index
        %c7 = arith.constant 7 : index
        %c1_20 = arith.constant 1 : index
        scf.for %arg15 = %c0_19 to %c7 step %c1_20 {
          %c1_21 = arith.constant 1 : index
          nvgpu.mbarrier.init %15[%arg15], %c1_21 : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7>
          %c1_22 = arith.constant 1 : index
          nvgpu.mbarrier.init %16[%arg15], %c1_22 : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7>
        }
        nvgpu.tma.prefetch.descriptor %4 : <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
        nvgpu.tma.prefetch.descriptor %5 : <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
      }
      scf.if %10 {
        nvvm.setmaxregister  decrease 40
        %true = arith.constant true
        %c0_19 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_20 = arith.constant 1 : index
        %18 = scf.for %arg15 = %c0_19 to %c16 step %c1_20 iter_args(%arg16 = %true) -> (i1) {
          %c7 = arith.constant 7 : index
          %19 = arith.remui %arg15, %c7 : index
          %c10000000 = arith.constant 10000000 : index
          nvgpu.mbarrier.try_wait.parity %15[%19], %arg16, %c10000000 : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7>
          %c6 = arith.constant 6 : index
          %20 = arith.cmpi eq, %19, %c6 : index
          %true_21 = arith.constant true
          %21 = arith.xori %arg16, %true_21 : i1
          %22 = arith.select %20, %21, %arg16 : i1
          %block_id_x = gpu.block_id  x
          %block_id_y = gpu.block_id  y
          %c128_22 = arith.constant 128 : index
          %23 = arith.muli %block_id_x, %c128_22 : index
          %c128_23 = arith.constant 128 : index
          %24 = arith.muli %block_id_y, %c128_23 : index
          %thread_id_x_24 = gpu.thread_id  x
          %c16384 = arith.constant 16384 : index
          %25 = arith.muli %19, %c16384 : index
          %c16384_25 = arith.constant 16384 : index
          %26 = arith.muli %19, %c16384_25 : index
          %c114688 = arith.constant 114688 : index
          %27 = arith.addi %26, %c114688 : index
          %c8192 = arith.constant 8192 : index
          %28 = arith.addi %27, %c8192 : index
          %29 = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
          %view = memref.view %29[%25][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<128x64xf16, #gpu.address_space<workgroup>>
          %30 = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
          %view_26 = memref.view %30[%27][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x64xf16, #gpu.address_space<workgroup>>
          %31 = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
          %view_27 = memref.view %31[%28][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x64xf16, #gpu.address_space<workgroup>>
          %c32768 = arith.constant 32768 : index
          nvgpu.mbarrier.arrive.expect_tx %16[%19], %c32768, predicate = %8 : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7>
          %c128_28 = arith.constant 128 : index
          %32 = arith.remui %thread_id_x_24, %c128_28 : index
          %c0_29 = arith.constant 0 : index
          %33 = arith.cmpi eq, %32, %c0_29 : index
          %c64_30 = arith.constant 64 : index
          %34 = arith.muli %arg15, %c64_30 : index
          nvgpu.tma.async.load %4[%34, %23], %16[%19] to %view, predicate = %33 : <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7> -> memref<128x64xf16, #gpu.address_space<workgroup>>
          nvgpu.tma.async.load %5[%24, %34], %16[%19] to %view_26, predicate = %33 : <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7> -> memref<64x64xf16, #gpu.address_space<workgroup>>
          %c64_31 = arith.constant 64 : index
          %35 = arith.addi %24, %c64_31 : index
          nvgpu.tma.async.load %5[%35, %34], %16[%19] to %view_27, predicate = %33 : <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7> -> memref<64x64xf16, #gpu.address_space<workgroup>>
          scf.yield %22 : i1
        }
      }
      scf.if %14 {
        nvvm.setmaxregister  increase 232
        %false = arith.constant false
        %18 = nvgpu.warpgroup.mma.init.accumulator -> <fragmented = vector<128x128xf32>>
        %c0_19 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_20 = arith.constant 1 : index
        %19:2 = scf.for %arg15 = %c0_19 to %c16 step %c1_20 iter_args(%arg16 = %18, %arg17 = %false) -> (!nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>>, i1) {
          %c7 = arith.constant 7 : index
          %23 = arith.remui %arg15, %c7 : index
          %c10000000 = arith.constant 10000000 : index
          nvgpu.mbarrier.try_wait.parity %16[%23], %arg17, %c10000000 : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7>
          %c16384 = arith.constant 16384 : index
          %24 = arith.muli %23, %c16384 : index
          %c114688 = arith.constant 114688 : index
          %25 = arith.addi %24, %c114688 : index
          %26 = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
          %view_28 = memref.view %26[%24][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<128x64xf16, #gpu.address_space<workgroup>>
          %27 = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
          %view_29 = memref.view %27[%25][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x128xf16, #gpu.address_space<workgroup>>
          %28 = nvgpu.warpgroup.generate.descriptor %view_28, %4 : memref<128x64xf16, #gpu.address_space<workgroup>>, <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none> -> <tensor = memref<128x64xf16, #gpu.address_space<workgroup>>>
          %29 = nvgpu.warpgroup.generate.descriptor %view_29, %5 : memref<64x128xf16, #gpu.address_space<workgroup>>, <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none> -> <tensor = memref<64x128xf16, #gpu.address_space<workgroup>>>
          %30 = nvgpu.warpgroup.mma %28, %29, %arg16 {transposeB} : <tensor = memref<128x64xf16, #gpu.address_space<workgroup>>>, <tensor = memref<64x128xf16, #gpu.address_space<workgroup>>>, <fragmented = vector<128x128xf32>> -> <fragmented = vector<128x128xf32>>
          %c0_30 = arith.constant 0 : index
          %31 = arith.cmpi ugt, %arg15, %c0_30 : index
          %32 = arith.andi %31, %12 : i1
          scf.if %32 {
            %c0_31 = arith.constant 0 : index
            %36 = arith.cmpi eq, %23, %c0_31 : index
            %c6_32 = arith.constant 6 : index
            %c1_33 = arith.constant 1 : index
            %37 = arith.subi %23, %c1_33 : index
            %38 = arith.select %36, %c6_32, %37 : index
            %39 = nvgpu.mbarrier.arrive %15[%38] : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7> -> !nvgpu.mbarrier.token
          }
          %c6 = arith.constant 6 : index
          %33 = arith.cmpi eq, %23, %c6 : index
          %true = arith.constant true
          %34 = arith.xori %arg17, %true : i1
          %35 = arith.select %33, %34, %arg17 : i1
          scf.yield %30, %35 : !nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>>, i1
        }
        nvvm.wgmma.wait.group.sync.aligned 0
        %thread_id_x_21 = gpu.thread_id  x
        %block_id_x = gpu.block_id  x
        %block_id_y = gpu.block_id  y
        %c128_22 = arith.constant 128 : index
        %20 = arith.muli %block_id_x, %c128_22 : index
        %c128_23 = arith.constant 128 : index
        %21 = arith.muli %block_id_y, %c128_23 : index
        %22 = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
        %c0_24 = arith.constant 0 : index
        %view = memref.view %22[%c0_24][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<128x128xf32, #gpu.address_space<workgroup>>
        %subview = memref.subview %memref_2[%20, %21] [128, 128] [1, 1] : memref<512x256xf32> to memref<128x128xf32, strided<[256, 1], offset: ?>>
        nvgpu.warpgroup.mma.store %19#0, %view : <fragmented = vector<128x128xf32>> to memref<128x128xf32, #gpu.address_space<workgroup>>
        gpu.barrier
        %c0_25 = arith.constant 0 : index
        %c128_26 = arith.constant 128 : index
        %c1_27 = arith.constant 1 : index
        scf.for %arg15 = %c0_25 to %c128_26 step %c1_27 {
          %23 = memref.load %view[%arg15, %thread_id_x_21] : memref<128x128xf32, #gpu.address_space<workgroup>>
          memref.store %23, %subview[%arg15, %thread_id_x_21] : memref<128x128xf32, strided<[256, 1], offset: ?>>
        }
      }
      gpu.terminator
    }
    %6 = gpu.memcpy async [%3] %arg2, %memref_2 : memref<512x256xf32>, memref<512x256xf32>
    gpu.wait [%6]
    return
  }
}

