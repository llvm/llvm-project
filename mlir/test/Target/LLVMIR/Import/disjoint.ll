; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK-LABEL: @disjointflag_inst
define void @disjointflag_inst(i64 %arg1, i64 %arg2) {
  ; CHECK: llvm.or disjoint %{{.*}}, %{{.*}} : i64
  %1 = or disjoint i64 %arg1, %arg2
  ret void
}
