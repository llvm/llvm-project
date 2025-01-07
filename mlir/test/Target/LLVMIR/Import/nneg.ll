; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK-LABEL: @nnegflag_inst
define void @nnegflag_inst(i32 %arg1) {
  ; CHECK: llvm.zext nneg %{{.*}} : i32 to i64
  %1 = zext nneg i32 %arg1 to i64
  ; CHECK: llvm.uitofp nneg %{{.*}} : i32 to f32
  %2 = uitofp nneg i32 %arg1 to float
  ret void
}
