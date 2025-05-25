; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK: llvm.func @tailkind()
declare void @tailkind()

; CHECK-LABEL: @call_tailkind
define void @call_tailkind() {
  ; CHECK: llvm.call musttail @tailkind()
  musttail call void @tailkind()
  ret void
}

; // -----

; CHECK: llvm.func @tailkind()
declare void @tailkind()

; CHECK-LABEL: @call_tailkind
define void @call_tailkind() {
  ; CHECK: llvm.call tail @tailkind()
  tail call void @tailkind()
  ret void
}

; // -----

; CHECK: llvm.func @tailkind()
declare void @tailkind()

; CHECK-LABEL: @call_tailkind
define void @call_tailkind() {
  ; CHECK: llvm.call notail @tailkind()
  notail call void @tailkind()
  ret void
}
