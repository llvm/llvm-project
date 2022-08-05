// RUN: mlir-opt %s -convert-gpu-to-rocdl -split-input-file | FileCheck %s
// RUN: mlir-opt %s \
// RUN:   -convert-gpu-to-rocdl=use-bare-ptr-memref-call-conv=true \
// RUN:   -split-input-file \
// RUN: | FileCheck %s --check-prefix=BARE

gpu.module @memref_conversions {
  // CHECK: llvm.func @kern
  // CHECK-SAME: (%{{.*}}: !llvm.ptr<f32>, %{{.*}}: !llvm.ptr<f32>, %{{.*}}: i64, %{{.*}}: i64, %{{.*}}: i64)
  // BARE: llvm.func @kern
  // BARE-SAME: (%{{.*}}: !llvm.ptr<f32>)
  gpu.func @kern(%arg0: memref<8xf32>) kernel {
    gpu.return
  }
}
