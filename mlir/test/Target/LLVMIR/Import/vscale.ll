; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

define void @vscale_func() vscale_range(2,8) {
  ; CHECK: llvm.func @vscale_func()
  ; CHECK-SAME: vscale_range(2, 8)
  ret void
}
