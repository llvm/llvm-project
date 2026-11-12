; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK-LABEL: llvm.func @frame_pointer_func
; CHECK-SAME: attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">}

define void @frame_pointer_func() "frame-pointer"="non-leaf" {
  ret void
}
