; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

define void @frame_pointer_func() "frame-pointer"="non-leaf" {
  ; CHECK: llvm.func @frame_pointer_func()
  ; CHECK-SAME: frame_pointer="non-leaf"
  ret void
}
