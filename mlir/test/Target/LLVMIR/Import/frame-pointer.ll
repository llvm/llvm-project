; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

define void @frame_pointer_func() "frame-pointer"="non-leaf" {
  ; CHECK: llvm.func @frame_pointer_func()
  ; CHECK-SAME: attributes {frame_pointer = 1 : i64}   
  
  ret void
}
