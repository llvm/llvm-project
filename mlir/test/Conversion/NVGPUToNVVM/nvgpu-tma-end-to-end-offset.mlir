// RUN: mlir-opt %s -convert-nvgpu-to-nvvm -expand-strided-metadata \
// RUN:   -finalize-memref-to-llvm -reconcile-unrealized-casts -canonicalize \
// RUN: | FileCheck %s

// End-to-end anchor for TMA async-load with a subview that produces a
// non-zero runtime offset (2 * 4096 = 8192). Pre-refactor, nvgpu-to-nvvm
// alone emitted `llvm.mlir.constant(8192)` because the static offset was
// baked into the memref type. Post-refactor, the offset is computed by
// memref-to-llvm from the subview indices `[2, 0, 0]` and stride 4096,
// so the 8192 constant only appears after the full pipeline runs.

!rhsTensorMap = !nvgpu.tensormap.descriptor<tensor = memref<64x64xf16, strided<[64, 1]>, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
!barrierType = !nvgpu.mbarrier.group<memorySpace = #gpu.address_space<workgroup>>

memref.global "private" @dynamicShmem : memref<0xf16,3>

// CHECK-LABEL: func @async_tma_load_subview
//       CHECK: arith.constant 8192 : index
func.func @async_tma_load_subview(%rhsTensorMap: !rhsTensorMap, %mbarrier: !barrierType) {
  %c0 = arith.constant 0 : index
  %dynamicMem = memref.get_global @dynamicShmem : memref<0xf16, 3>
  %rhsShmem2 = memref.reinterpret_cast %dynamicMem to offset: [0], sizes: [4, 64, 64], strides: [4096, 64, 1] : memref<0xf16, 3> to memref<4x64x64xf16,3>
  %rhsShmem3 = memref.subview %rhsShmem2[2, 0, 0][1, 64, 64][1, 1, 1] : memref<4x64x64xf16,3> to memref<1x64x64xf16, strided<[4096, 64, 1]>, 3>
  %rhsShmem = memref.subview %rhsShmem3[0, 0, 0][1, 64, 64][1, 1, 1] : memref<1x64x64xf16, strided<[4096, 64, 1]>, 3> to memref<64x64xf16, strided<[64, 1]>, 3>
  nvgpu.tma.async.load %rhsTensorMap[%c0, %c0], %mbarrier[%c0] to %rhsShmem : !rhsTensorMap, !barrierType -> memref<64x64xf16, strided<[64, 1]>, 3>
  return
}
