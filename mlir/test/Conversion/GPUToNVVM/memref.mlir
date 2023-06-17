// RUN: mlir-opt %s -convert-gpu-to-nvvm | FileCheck %s
// RUN: mlir-opt %s -convert-gpu-to-nvvm='use-bare-ptr-memref-call-conv=1' \
// RUN: | FileCheck %s --check-prefix=BARE

gpu.module @memref_conversions {
  // CHECK: llvm.func @kern
  // CHECK-SAME: (%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: i64, %{{.*}}: i64, %{{.*}}: i64)
  // BARE: llvm.func @kern
  // BARE-SAME: (%{{.*}}: !llvm.ptr)
  gpu.func @kern(%arg0: memref<8xf32>) kernel {
    gpu.return
  }
}
