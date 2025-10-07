// RUN: mlir-opt %s -cse -convert-vector-to-llvm="enable-amx" | FileCheck %s

// With inclusion of memory side-effects, it is expected CSE not to fold multiple 
// "tileload" and "tilezero".
// CHECK-LABEL: do_not_fold_tiles(
// CHECK: llvm.call_intrinsic "llvm.x86.tilezero.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tilezero.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tileloadd64.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tileloadd64.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tileloadd64.internal"
// CHECK: llvm.call_intrinsic "llvm.x86.tileloadd64.internal"
func.func @do_not_fold_tiles(%arg0: memref<2x32x32xbf16>, %arg1: memref<2x16x32xbf16>) -> memref<16x32xf32> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c16 = arith.constant 16 : index
  %alloca = memref.alloca() : memref<16x32xf32>
  %0 = amx.tile_zero : !amx.tile<16x16xf32>
  %1 = amx.tile_zero : !amx.tile<16x16xf32>
  %2:2 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %0, %arg4 = %1) -> (!amx.tile<16x16xf32>, !amx.tile<16x16xf32>) {
    %3 = amx.tile_load %arg0[%arg2, %c0, %c0] : memref<2x32x32xbf16> into !amx.tile<16x32xbf16>
    %4 = amx.tile_load %arg0[%arg2, %c16, %c0] : memref<2x32x32xbf16> into !amx.tile<16x32xbf16>
    %5 = amx.tile_load %arg1[%arg2, %c0, %c0] : memref<2x16x32xbf16> into !amx.tile<16x32xbf16>
    %6 = amx.tile_load %arg1[%arg2, %c0, %c0] : memref<2x16x32xbf16> into !amx.tile<16x32xbf16>
    %7 = amx.tile_mulf %3, %5, %arg3 : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
    %8 = amx.tile_mulf %4, %6, %arg4 : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
    scf.yield %7, %8 : !amx.tile<16x16xf32>, !amx.tile<16x16xf32>
  }
  amx.tile_store %alloca[%c0, %c0], %2#0 : memref<16x32xf32>, !amx.tile<16x16xf32>
  amx.tile_store %alloca[%c0, %c16], %2#1 : memref<16x32xf32>, !amx.tile<16x16xf32>
  return %alloca : memref<16x32xf32>
}
