; RUN: mlir-translate --import-llvm %s | FileCheck %s

; CHECK: llvm.func @shufflevector_crash
define void @shufflevector_crash(<2 x ptr> %arg0) {
  ; CHECK: llvm.shufflevector %{{.+}}, %{{.+}} [1, 0] : !llvm.vec<2 x ptr>
  %1 = shufflevector <2 x ptr> %arg0, <2 x ptr> undef, <2 x i32> <i32 1, i32 0>
  ret void
}
