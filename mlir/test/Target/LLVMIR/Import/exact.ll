; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK-LABEL: @exactflag_inst
define void @exactflag_inst(i64 %arg1, i64 %arg2) {
  ; CHECK: llvm.udiv exact %{{.*}}, %{{.*}} : i64
  %1 = udiv exact i64 %arg1, %arg2
  ; CHECK: llvm.sdiv exact %{{.*}}, %{{.*}} : i64
  %2 = sdiv exact i64 %arg1, %arg2
  ; CHECK: llvm.lshr exact %{{.*}}, %{{.*}} : i64
  %3 = lshr exact i64 %arg1, %arg2
  ; CHECK: llvm.ashr exact %{{.*}}, %{{.*}} : i64
  %4 = ashr exact i64 %arg1, %arg2
  ret void
}
