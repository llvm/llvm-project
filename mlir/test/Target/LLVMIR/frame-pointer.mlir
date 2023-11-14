// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @frame_pointer_func() frame_pointer="non-leaf" {
  // CHECK-LABEL: define void @frame_pointer_func() #[[ATTRS:\d+]]
  // CHECK: attributes #[[ATTRS]] = { "frame-pointer"="non-leaf" }
  llvm.return
}

