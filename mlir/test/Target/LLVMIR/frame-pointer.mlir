// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define void @frame_pointer_func() 
// CHECK-SAME: #[[ATTRS:[0-9]+]]
llvm.func @frame_pointer_func() attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">}  {
  llvm.return
}
// CHECK: attributes #[[ATTRS]] = { "frame-pointer"="non-leaf" }
