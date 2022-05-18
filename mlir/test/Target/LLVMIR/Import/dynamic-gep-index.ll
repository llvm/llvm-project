; RUN: mlir-translate --import-llvm %s | FileCheck %s

%sub_struct = type { i32, i8 }
%my_struct = type { %sub_struct, [4 x i32] }

; CHECK: llvm.func @foo(%[[ARG0:.+]]: !llvm.ptr<struct<"my_struct", {{.+}}>>, %[[ARG1:.+]]: i32)
define void @foo(%my_struct* %arg, i32 %idx) {
  ; CHECK: %[[C0:.+]] = llvm.mlir.constant(0 : i32)
  ; CHECK: llvm.getelementptr %[[ARG0]][%[[C0]], 1, %[[ARG1]]]
  %1 = getelementptr %my_struct, %my_struct* %arg, i32 0, i32 1, i32 %idx
  ret void
}
