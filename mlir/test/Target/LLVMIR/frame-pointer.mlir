// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @frame_pointer_func() attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">}  {
  // CHECK-LABEL: define void @frame_pointer_func() 
  // CHECK-SAME: #[[ATTRS:[0-9]+]]
  // CHECK: attributes #[[ATTRS]] = { "frame-pointer"="non-leaf" }
  llvm.return
}

