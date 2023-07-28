// RUN: mlir-opt %s --convert-nvgpu-to-nvvm -gpu-kernel-outlining \
// RUN:     -convert-scf-to-cf -convert-nvvm-to-llvm \
// RUN:     -convert-vector-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -expand-strided-metadata \
// RUN:     -lower-affine \
// RUN:     -convert-index-to-llvm=index-bitwidth=32 \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-func-to-llvm \
// RUN:     -canonicalize \
// RUN: | mlir-opt -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-nvvm,convert-nvgpu-to-nvvm{use-opaque-pointers=1},lower-affine,convert-scf-to-cf,convert-vector-to-llvm,convert-math-to-llvm,expand-strided-metadata,lower-affine,convert-index-to-llvm{index-bitwidth=32},convert-arith-to-llvm,reconcile-unrealized-casts,gpu-to-cubin{chip=sm_90 features=+ptx80 dump-ptx}))' \
// RUN: 2&>1 | FileCheck %s --check-prefixes=CHECK-PTX

// CHECK-PTX: mbarrier.init.shared.b64
// CHECK-PTX: mbarrier.arrive.expect_tx.shared.b64
// CHECK-PTX: cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
// CHECK-PTX: cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
// CHECK-PTX: mbarrier.arrive.expect_tx.shared.b64
// CHECK-PTX: mbarrier.try_wait.parity.shared.b64

module @mymod {
  memref.global "private" @bufferLhsGlobal : memref<64x8xf32, 3>
  memref.global "private" @bufferRhsGlobal : memref<8x128xf32, 3>
  func.func @main() {
    %c10000000 = arith.constant 10000000 : index
    %c6144 = arith.constant 6144 : index
    %c45 = arith.constant 45 : index
    %c7 = arith.constant 7 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant 3.000000e+00 : f32
    %alloc = memref.alloc() : memref<64x8xf32>
    %alloc_0 = memref.alloc() : memref<8x128xf32>
    scf.for %arg0 = %c0 to %c8 step %c1 {
      scf.for %arg1 = %c0 to %c128 step %c1 {
        memref.store %cst, %alloc_0[%arg0, %arg1] : memref<8x128xf32>
      }
    }
    scf.for %arg0 = %c0 to %c64 step %c1 {
      scf.for %arg1 = %c0 to %c8 step %c1 {
        %5 = arith.index_cast %arg1 : index to i64
        %6 = arith.uitofp %5 : i64 to f32
        memref.store %6, %alloc[%arg0, %arg1] : memref<64x8xf32>
      }
    }
    %0 = gpu.wait async
    %memref, %asyncToken = gpu.alloc async [%0] () : memref<64x8xf32>
    %memref_1, %asyncToken_2 = gpu.alloc async [%0] () : memref<8x128xf32>
    %1 = gpu.memcpy async [%0] %memref, %alloc : memref<64x8xf32>, memref<64x8xf32>
    %2 = gpu.memcpy async [%0] %memref_1, %alloc_0 : memref<8x128xf32>, memref<8x128xf32>
    %cast = memref.cast %memref : memref<64x8xf32> to memref<*xf32>
    %cast_3 = memref.cast %memref_1 : memref<8x128xf32> to memref<*xf32>
    %3 = nvgpu.tma.create.descriptor %cast box[%c64, %c8] : memref<*xf32> -> <tensor = memref<64x8xf32, 3>, swizzle = none, l2promo = none, oob = zero, interleave = none>
    %4 = nvgpu.tma.create.descriptor %cast_3 box[%c8, %c128] : memref<*xf32> -> <tensor = memref<8x128xf32, 3>, swizzle = none, l2promo = none, oob = zero, interleave = none>
    gpu.launch blocks(%arg0, %arg1, %arg2) in (%arg6 = %c1, %arg7 = %c1, %arg8 = %c1) threads(%arg3, %arg4, %arg5) in (%arg9 = %c128, %arg10 = %c1, %arg11 = %c1) {
      %5 = gpu.block_dim  x
      %6 = gpu.thread_id  x
      %7 = memref.get_global @bufferLhsGlobal : memref<64x8xf32, 3>
      %8 = memref.get_global @bufferRhsGlobal : memref<8x128xf32, 3>
      %9 = nvgpu.mbarrier.create -> <memorySpace = #gpu.address_space<workgroup>>
      nvgpu.mbarrier.init %9, %5 : <memorySpace = #gpu.address_space<workgroup>>
      gpu.barrier
      %10 = arith.cmpi eq, %6, %c0 : index
      scf.if %10 {
        nvgpu.mbarrier.arrive.expect_tx %9, %c6144 : <memorySpace = #gpu.address_space<workgroup>>
        %11 = memref.load %7[%c0, %c0] : memref<64x8xf32, 3>
        %12 = memref.load %8[%c0, %c0] : memref<8x128xf32, 3>
        gpu.printf "[GPU] TMA BEFORE lhs[45][7] %f\0A" %11 : f32
        gpu.printf "[GPU] TMA BEFORE rhs[7][0] %f\0A" %12 : f32
        nvgpu.tma.async.load %3[%c0, %c0], %9 to %7 : <tensor = memref<64x8xf32, 3>, swizzle = none, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>> -> memref<64x8xf32, 3>
        nvgpu.tma.async.load %4[%c0, %c0], %9 to %8 : <tensor = memref<8x128xf32, 3>, swizzle = none, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>> -> memref<8x128xf32, 3>
      } else {
        nvgpu.mbarrier.arrive.expect_tx %9, %c0 : <memorySpace = #gpu.address_space<workgroup>>
      }
      nvgpu.mbarrier.try_wait.parity %9, %c0, %c10000000 : <memorySpace = #gpu.address_space<workgroup>>
      scf.if %10 {
        %11 = memref.load %7[%c45, %c7] : memref<64x8xf32, 3>
        %12 = memref.load %8[%c7, %c0] : memref<8x128xf32, 3>
        gpu.printf "[GPU] TMA LOADED lhs[45][7] %f\0A" %11 : f32
        gpu.printf "[GPU] TMA LOADED rhs[7][0] %f\0A" %12 : f32
      }
      gpu.terminator
    }
    return
  }
}