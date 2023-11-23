; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK-LABEL: @intflag_inst
define void @intflag_inst(i64 %arg1, i64 %arg2) {
  ; CHECK: llvm.add %{{.*}}, %{{.*}} flags <nsw> : i64
  %1 = add nsw i64 %arg1, %arg2
  ; CHECK: llvm.sub %{{.*}}, %{{.*}} flags <nuw> : i64
  %2 = sub nuw i64 %arg1, %arg2
  ; CHECK: llvm.mul %{{.*}}, %{{.*}} flags <nsw, nuw> : i64
  %3 = mul nsw nuw i64 %arg1, %arg2
  ; CHECK: llvm.shl %{{.*}}, %{{.*}} flags <nsw, nuw> : i64
  %4 = shl nuw nsw i64 %arg1, %arg2
  ret void
}
