; RUN: mlir-translate --import-llvm %s | FileCheck %s

; Ensure both structs have different names.
; CHECK: llvm.func @fn(!llvm.struct<"[[NAME:[^"]*]]",
; CHECK-NOT: struct<"[[NAME]]",
%0 = type { %1 }
%1 = type { i8 }
declare void @fn(%0)
