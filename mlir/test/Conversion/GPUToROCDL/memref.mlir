// RUN: mlir-opt %s -convert-gpu-to-rocdl='use-opaque-pointers=1' -split-input-file | FileCheck %s
// RUN: mlir-opt %s \
// RUN:   -convert-gpu-to-rocdl='use-bare-ptr-memref-call-conv=true use-opaque-pointers=1' \
// RUN:   -split-input-file \
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
