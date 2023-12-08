// RUN: mlir-opt %s \
// RUN:  -gpu-lower-to-nvvm="cubin-chip=sm_90a cubin-features=+ptx80 opt-level=3" \
// RUN:  | mlir-cpu-runner \
// RUN:   --shared-libs=%mlir_cuda_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --entry-point-result=void \
// RUN:  | FileCheck %s

// CHECK: Correct Results :
// CHECK: 16384
// CHECK: Incorrect Results :
// CHECK: 0

// This program performs 128x128x128 GEMM (F32 += F16 * F16)
//
// ## Sequential
// for(128)
//  for(128)
//   for(128)
//    D += A * B
//
// ## Parallel 1 CTA with 1 Warpgroup with 2 pipelining stage
//
//  cuda kernel() {
//    mbarriers.init[2]
//    for(i = 0;...2) {
//       tma.load shmem_buffer<i x...>
//       mbarrier.expect_tx group[i]
//    }
//    result = 
//      for(i = 0;...2) {
//        pipe = i % 2
//        mbarrier.wait [pipe]
//        lhs = shmem_buffer_lhs<pipe x 128 x 64>
//        rhs = shmem_buffer_rhs<pipe x 64 x 128>
//        yield nvgpu.warpgroup.mma (lhs, rhs)
//        ---------------------------------------------------------------------
//        Expanded : nvgpu.warpgroup.mma [128][128]+=[128][64]*[64][128]
//                       wgmma.m64n128k16(A[0:64][0:16]  *  B[0:16][0:128])
//                       wgmma.m64n128k16(A[0:64][16:32] *  B[16:32][0:128])
//                       wgmma.m64n128k16(A[0:64][32:48] *  B[32:48][0:128])
//                       wgmma.m64n128k16(A[0:64][48:64] *  B[48:64][0:128])
//                       wgmma.m64n128k16(A[64:128][0:16]  *  B[0:16][0:128])
//                       wgmma.m64n128k16(A[64:128][16:32] *  B[16:32][0:128])
//                       wgmma.m64n128k16(A[64:128][32:48] *  B[32:48][0:128])
//                       wgmma.m64n128k16(A[64:128][48:64] *  B[48:64][0:128])
//        ---------------------------------------------------------------------
//      }
//    nvgpu.store result -> shmem_buffer_result


!barrierType = !nvgpu.mbarrier.group<memorySpace = #gpu.address_space<workgroup>, num_barriers = 2>
!lhsTensorMap = !nvgpu.tensormap.descriptor<tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo=none, oob=zero, interleave=none>
!rhsTensorMap = !nvgpu.tensormap.descriptor<tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo=none, oob=zero, interleave=none>

func.func private @printMemrefF32(memref<*xf32>)

memref.global "private" @dynamicShmem : memref<0xf16, 3> {alignment = 16 : i64}
memref.global "private" @accShmem : memref<0xf32, 3> {alignment = 16 : i64}

func.func @main() {
  // matrix A (128*64) * matrix B (64*128) * stages(2)
  // matrix A [128][64] * matrix B[64][128] * stages(2)
  %shmemSize = arith.constant 65536 : i32
  %hc1 = arith.constant 1 : index
  %hc4096 = arith.constant 4096 : index
  %hc0 = arith.constant 0 : index
  %hc64 = arith.constant 64 : index
  %hc16 = arith.constant 16 : index
  %hc8 = arith.constant 8 : index
  %hc128 = arith.constant 128 : index
  %hc32 = arith.constant 32 : index
  %hc256 = arith.constant 256 : index
  %f0 = arith.constant 0.0 : f32

  // Step 1. Allocate and Initilize LHS and RHS Matrices 
  %matrixAHost = memref.alloc() : memref<128x128xf16>
  %matrixBHost = memref.alloc() : memref<128x128xf16>
  %matrixDHost = memref.alloc() : memref<128x128xf32>
  %matrixRefHost = memref.alloc() : memref<128x128xf32>
  scf.for %i = %hc0 to %hc128 step %hc1 {
    scf.for %j = %hc0 to %hc128 step %hc1 {
      %v0 = arith.muli %i, %hc128 : index         // i * 128
      %v00 = arith.addi %v0, %j : index           // i * 128 + j
      %v01 = arith.divui %v00, %hc8 : index        // (i * 128 + j) / 8
      %v02 = arith.remui %v01, %hc16 : index      // <<<<< mod 128
      %v2 = arith.index_cast %v02 : index to i32
      %vR = arith.sitofp %v2 : i32 to f16
      memref.store %vR, %matrixBHost[%i, %j] : memref<128x128xf16>
      %b0 = arith.muli %j, %hc64 : index
      %b00 = arith.addi %b0, %i : index
      %b01 = arith.divui %b00, %hc8 : index
      %b02 = arith.remui %b01, %hc16 : index      // <<<<< mod 128
      %v1 = arith.index_cast %b02 : index to i32
      %vL = arith.sitofp %v1 : i32 to f16
      memref.store %vL, %matrixAHost[%j, %i] : memref<128x128xf16>
      memref.store %f0, %matrixDHost[%i, %j] : memref<128x128xf32>
      memref.store %f0, %matrixRefHost[%i, %j] : memref<128x128xf32>
    }
  }

  // Step 2. Allocate Device Memory for LHS and RHS Matrices and Copy H2D
  %token = gpu.wait async
  %matrixA:2 = gpu.alloc async [%token] () : memref<128x128xf16>
  %matrixB:2 = gpu.alloc async [%token]  () : memref<128x128xf16>
  %matrixD:2 = gpu.alloc async [%token] () : memref<128x128xf32>
  %1 = gpu.memcpy async [%token] %matrixA, %matrixAHost : memref<128x128xf16>, memref<128x128xf16>
  %2 = gpu.memcpy async [%token] %matrixB, %matrixBHost : memref<128x128xf16>, memref<128x128xf16>
  %castA = memref.cast %matrixA : memref<128x128xf16> to memref<*xf16>
  %castB = memref.cast %matrixB : memref<128x128xf16> to memref<*xf16>

  // Step 3. Create TMA Descriptor
  %descA = nvgpu.tma.create.descriptor %castA box[%hc128, %hc64] : memref<*xf16> -> !lhsTensorMap
  %descB = nvgpu.tma.create.descriptor %castB box[%hc64, %hc64] : memref<*xf16> -> !rhsTensorMap

  // Step 4. Launch GPU Kernel
  gpu.launch blocks(%arg0, %arg1, %arg2) in (%arg6 = %hc1, %arg7 = %hc1, %arg8 = %hc1) 
            threads(%arg3, %arg4, %arg5) in (%arg9 = %hc128, %arg10 = %hc1, %arg11 = %hc1) 
            dynamic_shared_memory_size %shmemSize 
  {  
    memref.assume_alignment %matrixD, 16 : memref<128x128xf32>

    %c256 = arith.constant 256 : index
    %c10000000 = arith.constant 10000000 : index
    %c32768 = arith.constant 32768 : index
    %c320 = arith.constant 320 : index
    %c192 = arith.constant 192 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c7 = arith.constant 7 : index    
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c4096 = arith.constant 4096 : index
    %c8 = arith.constant 8 : index
    %txcount = arith.constant 32768 : index     

    %tidx = gpu.thread_id  x
    %dynamicMem = memref.get_global @dynamicShmem : memref<0xf16, 3>
    %lhsShmem = memref.reinterpret_cast %dynamicMem to offset: [0], sizes: [2, 128, 64], strides: [8192, 64, 1] : memref<0xf16, 3> to memref<2x128x64xf16, 3>
    %rhsShmem2 = memref.reinterpret_cast %dynamicMem to offset: [0], sizes: [4, 64, 128],  strides: [8192,128,1] : memref<0xf16, 3> to memref<4x64x128xf16,3>
    %rhsShmem = memref.subview %rhsShmem2[2, 0, 0][2, 64, 128][1, 1, 1] : memref<4x64x128xf16,3> to memref<2x64x128xf16, strided<[8192, 128, 1], offset: 16384>, 3>
    
    // Step 1. [GPU] Create Async Transactional Barriers (mbarriers)
    %barrier = nvgpu.mbarrier.create -> !barrierType
    %cnd = arith.cmpi eq, %tidx, %c0 : index

    // Step 2. [GPU] Initialize mbarriers 
    nvgpu.mbarrier.init %barrier[%c0], %c1 : !barrierType
    nvgpu.mbarrier.init %barrier[%c1], %c1 : !barrierType
    
    // Step 3. [GPU] Prefetch TMA Descriptors to L1 Cache
    nvgpu.tma.prefetch.descriptor %descA : !lhsTensorMap
    nvgpu.tma.prefetch.descriptor %descB : !rhsTensorMap
    
    // Step 4.1 [GPU] TMA Load Pipeline 1   
    scf.if %cnd {
      %pipe = arith.constant 0 : index
      %lhsSlice = memref.subview %lhsShmem[0, 0, 0][1, 128, 64][1, 1, 1] : memref<2x128x64xf16, 3> to memref<128x64xf16, 3>
      %rhsSlice = memref.subview %rhsShmem[0, 0, 0][1, 64, 128][1, 1, 1] : memref<2x64x128xf16, strided<[8192, 128, 1], offset: 16384>, 3> to memref<64x128xf16, strided<[128, 1], offset: 16384>, 3>
      %halfFirst = memref.subview %rhsSlice[0, 0][64, 64][1, 1] : memref<64x128xf16, strided<[128, 1], offset: 16384>, 3> to memref<64x64xf16, strided<[128, 1], offset: 16384>, 3>
      %halfSecond = memref.subview %rhsSlice[32, 0][64, 64][1, 1] : memref<64x128xf16, strided<[128, 1], offset: 16384>, 3> to memref<64x64xf16, strided<[128, 1], offset: 20480>, 3>
      nvgpu.mbarrier.arrive.expect_tx %barrier[%pipe], %txcount : !barrierType        
      %dim = arith.muli %pipe, %c64 : index
      nvgpu.tma.async.load %descA[%dim, %c0], %barrier[%pipe] to %lhsSlice : !lhsTensorMap, !barrierType -> memref<128x64xf16, 3>
      nvgpu.tma.async.load %descB[%c0, %dim], %barrier[%pipe] to %halfFirst : !rhsTensorMap, !barrierType -> memref<64x64xf16, strided<[128, 1], offset: 16384>, 3>
      nvgpu.tma.async.load %descB[%c64, %dim], %barrier[%pipe] to %halfSecond : !rhsTensorMap, !barrierType -> memref<64x64xf16, strided<[128, 1], offset: 20480>, 3>
    }
    // Step 4.2 [GPU] TMA Load Pipeline 2
    scf.if %cnd {
      %pipe = arith.constant 1 : index
      %lhsSlice = memref.subview %lhsShmem[1, 0, 0][1, 128, 64][1, 1, 1] : memref<2x128x64xf16, 3> to memref<128x64xf16, strided<[64, 1], offset: 8192>, 3>
      %rhsSlice = memref.subview %rhsShmem[1, 0, 0][1, 64, 128][1, 1, 1] : memref<2x64x128xf16, strided<[8192, 128, 1], offset: 16384>, 3> to memref<64x128xf16, strided<[128, 1], offset: 24576>, 3>
      %halfFirst = memref.subview %rhsSlice[0, 0][64, 64][1, 1] : memref<64x128xf16, strided<[128, 1], offset: 24576>, 3> to memref<64x64xf16, strided<[128, 1], offset: 24576>, 3>
      %halfSecond = memref.subview %rhsSlice[32, 0][64, 64][1, 1] : memref<64x128xf16, strided<[128, 1], offset: 24576>, 3> to memref<64x64xf16, strided<[128, 1], offset: 28672>, 3>
      nvgpu.mbarrier.arrive.expect_tx %barrier[%pipe], %txcount : !barrierType
      %dim = arith.muli %pipe, %c64 : index  
      nvgpu.tma.async.load %descA[%dim, %c0], %barrier[%pipe] to %lhsSlice : !lhsTensorMap, !barrierType -> memref<128x64xf16, strided<[64, 1], offset: 8192>, 3>
      nvgpu.tma.async.load %descB[%c0, %dim], %barrier[%pipe] to %halfFirst : !rhsTensorMap, !barrierType -> memref<64x64xf16, strided<[128, 1], offset: 24576>, 3>
      nvgpu.tma.async.load %descB[%c64, %dim], %barrier[%pipe] to %halfSecond : !rhsTensorMap, !barrierType -> memref<64x64xf16, strided<[128, 1], offset: 28672>, 3>
    }
    
    // Step 5. [GPU] Initiliaze accumulator matrix
    %14 = nvgpu.warpgroup.mma.init.accumulator -> <fragmented = vector<128x128xf32>>

    // Step 6. [GPU] Main Loop Starts
    %15 = scf.for %i = %c0 to %c2 step %c1 iter_args(%mc = %14) 
                    -> (!nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>>)
    {
      %ticks = arith.constant 10000000 : index
      // TMA wait
      nvgpu.mbarrier.try_wait.parity %barrier[%i], %c0, %ticks : !barrierType
      %lhsSlice = memref.subview %lhsShmem [%i, 0, 0][1, 128, 64][1, 1, 1] : memref<2x128x64xf16, 3> to memref<128x64xf16, strided<[64, 1], offset: ?>, 3>
      %rhsSlice = memref.subview %rhsShmem [%i, 0, 0][1, 64, 128][1, 1, 1] : memref<2x64x128xf16, strided<[8192, 128, 1], offset: 16384>, 3> to memref<64x128xf16, strided<[128, 1], offset: ?>, 3>
      // Descriptor WGMMA
      %dA = nvgpu.warpgroup.generate.descriptor %lhsSlice, %descA : memref<128x64xf16, strided<[64, 1], offset: ?>, 3>, !lhsTensorMap -> !nvgpu.warpgroup.descriptor<tensor=memref<128x64xf16, 3>>
      %dB = nvgpu.warpgroup.generate.descriptor %rhsSlice, %descB : memref<64x128xf16, strided<[128, 1], offset: ?>, 3>, !rhsTensorMap -> !nvgpu.warpgroup.descriptor<tensor=memref<64x128xf16, 3>>
      // Perform WGMMA 128x128x64
      %md  = nvgpu.warpgroup.mma %dA, %dB, %mc {transposeB} : <tensor = memref<128x64xf16,3>>, <tensor = memref<64x128xf16,3>>, <fragmented = vector<128x128xf32>> -> <fragmented = vector<128x128xf32>>
      scf.yield %md : !nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>>
    }
    
    // Step 7. Wait all to finish mma
    nvvm.wgmma.wait.group.sync.aligned 0

    // Step 8. [GPU] Epilogue, store fragmented register to shared memory
    %accShmem = memref.get_global @accShmem : memref<0xf32, 3>
    %accShmemPtr = memref.reinterpret_cast %accShmem to offset: [0], sizes: [128, 128], strides: [128, 1] : memref<0xf32, 3> to memref<128x128xf32, 3>
    nvgpu.warpgroup.mma.store %15, %accShmemPtr : <fragmented = vector<128x128xf32>> to memref<128x128xf32, 3>
    
    // Step 9. [GPU] Epilogue, shared memory to global memory
    %17 = arith.divui %tidx, %c32 : index
    %18 = arith.remui %tidx, %c32 : index
    scf.for %arg12 = %17 to %c128 step %c4 {
      %19 = arith.muli %18, %c4 : index
      %20 = vector.load %accShmemPtr[%arg12, %19] : memref<128x128xf32, 3>, vector<4xf32>
      vector.store %20, %matrixD[%arg12, %19] : memref<128x128xf32>, vector<4xf32>
    }
    gpu.terminator
  }

  // Step 5. Copy D2H
  %5 = gpu.memcpy async [%token] %matrixDHost, %matrixD  : memref<128x128xf32>, memref<128x128xf32>
  gpu.wait [%token]

  // Step 6. Compute on host
  linalg.matmul ins(%matrixAHost, %matrixBHost : memref<128x128xf16>, memref<128x128xf16>) outs(%matrixRefHost : memref<128x128xf32>)
  
  // Step 7. Verify
  %ic1 = arith.constant 1 : i32
  %ic0 = arith.constant 0 : i32
  %tolerance = arith.constant 0.00000001 : f32
  %errorCount, %correctCount = 
  scf.for %i = %hc0 to %hc128 step %hc1 iter_args(%ec1 = %ic0, %cc1 = %ic0) -> (i32,i32) {
    %ec2, %cc2 = 
    scf.for %j = %hc0 to %hc128 step %hc1  iter_args(%ec2 = %ec1, %cc2 = %cc1) -> (i32,i32){
      %v1 = memref.load %matrixRefHost[%i,%j] : memref<128x128xf32>
      %v2 = memref.load %matrixDHost[%i,%j] : memref<128x128xf32>
      %g1 = arith.subf %v1,%v2 : f32
      %g2 = math.absf %g1: f32
      %g3 = arith.cmpf ult, %tolerance, %g2 : f32        
      %ec3, %cc3 = scf.if %g3 -> (i32, i32) {
        %coor = arith.constant dense<-1> : vector<2xi32>
        %i32 = arith.index_cast %i : index to i32
        %j32 = arith.index_cast %j : index to i32
        %coord1 = vector.insert %i32, %coor[0] : i32 into vector<2xi32>
        %coord2 = vector.insert %j32, %coord1[1] : i32 into vector<2xi32>        
        %ec3 = arith.addi %ec2, %ic1 : i32
        scf.yield %ec3, %cc2 : i32, i32
      } else {
        %cc3 = arith.addi %cc2, %ic1 : i32
        scf.yield %ec2, %cc3 : i32, i32
      }
      scf.yield %ec3, %cc3 : i32,i32
    }
    scf.yield %ec2,%cc2 : i32,i32
  }

  vector.print str "Correct Results :"
  vector.print %correctCount : i32
  vector.print str "Incorrect Results :"
  vector.print %errorCount : i32

  return
}
