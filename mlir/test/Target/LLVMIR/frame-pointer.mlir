// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @frame_pointer_func() attributes {frame_pointer = 1 : i64}  {
  // CHECK-LABEL: define void @frame_pointer_func() 
  // CHECK-SAME: #[[ATTRS:[0-9]+]]
  // CHECK: attributes #[[ATTRS]] = { "frame-pointer"="non-leaf" }
  llvm.return
}

