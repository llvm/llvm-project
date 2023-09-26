// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @vscale_func() vscale_range(2,8) {
  // CHECK-LABEL: define void @vscale_func
  // CHECK: attributes #{{.*}} = { vscale_range(2,8) }
  llvm.return
}
