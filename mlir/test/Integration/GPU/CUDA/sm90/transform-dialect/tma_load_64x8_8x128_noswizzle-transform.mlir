// RUN: mlir-opt %s \
// RUN:     -transform-interpreter \
// RUN:     -test-transform-dialect-erase-schedule \
// RUN:     -convert-nvgpu-to-nvvm -gpu-kernel-outlining \
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
// RUN:     -expand-strided-metadata --nvvm-attach-target="module=main_kernel features=+ptx80 chip=sm_90 O=3" \
// RUN:  | mlir-opt -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-nvvm,convert-index-to-llvm{index-bitwidth=32},canonicalize,cse))' \
// RUN:  | mlir-opt --gpu-to-llvm --gpu-module-to-binary=format=%gpu_compilation_format -canonicalize -cse -reconcile-unrealized-casts -debug-only=serialize-to-isa \
// RUN: 2>&1 | FileCheck %s --check-prefixes=CHECK-PTX

// Basic PTX check to make sure we are generating the right instructions.
// CHECK-PTX: mbarrier.init.shared.b64
// CHECK-PTX: mbarrier.arrive.expect_tx.shared.b64
// CHECK-PTX: cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
// CHECK-PTX: cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
// CHECK-PTX: mbarrier.arrive.expect_tx.shared.b64
// CHECK-PTX: mbarrier.try_wait.parity.shared.b64

// RUN: mlir-opt %s \
// RUN:     -transform-interpreter \
// RUN:     -test-transform-dialect-erase-schedule \
// RUN:     -convert-nvgpu-to-nvvm -gpu-kernel-outlining \
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
// RUN:     -expand-strided-metadata --nvvm-attach-target="module=main_kernel features=+ptx80 chip=sm_90 O=3" \
// RUN:  | mlir-opt -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-nvvm,convert-index-to-llvm{index-bitwidth=32},canonicalize,cse))' \
// RUN:  | mlir-opt --gpu-to-llvm --gpu-module-to-binary=format=%gpu_compilation_format -canonicalize -cse -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%mlir_cuda_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// CHECK: [GPU] TMA LOADED lhs[45][7] 7.000000
// CHECK: [GPU] TMA LOADED rhs[7][0] 3.000000

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

  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
            threads(%tx, %ty, %tz) in (%block_x = %c128, %block_y = %c1, %block_z = %c1) {
    %out = memref.get_global @bufferLhsGlobal : memref<64x8xf32, 3>
    %out_1 = memref.get_global @bufferRhsGlobal : memref<8x128xf32, 3>
    linalg.copy ins(%memref: memref<64x8xf32>) outs(%out: memref<64x8xf32, 3>)
    linalg.copy ins(%memref_1: memref<8x128xf32>) outs(%out_1: memref<8x128xf32, 3>)

    %6 = gpu.thread_id  x
    %10 = arith.cmpi eq, %6, %c0 : index
    scf.if %10 {
      %11 = memref.load %out[%c45, %c7] : memref<64x8xf32, 3>
      %12 = memref.load %out_1[%c7, %c0] : memref<8x128xf32, 3>
      gpu.printf "[GPU] TMA LOADED lhs[45][7] %f\0A", %11 : f32
      gpu.printf "[GPU] TMA LOADED rhs[7][0] %f\0A", %12 : f32
    }
    gpu.terminator
  }

  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %copy = transform.structured.match ops{["linalg.copy"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    transform.nvgpu.rewrite_copy_as_tma %copy
      : (!transform.any_op) -> ()
      transform.yield
  }
}
