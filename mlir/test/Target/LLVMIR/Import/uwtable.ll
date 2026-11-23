; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK-LABEL: llvm.func @uwtable_func
; CHECK-SAME: attributes {uwtable_kind = #llvm.uwtableKind<sync>}
define void @uwtable_func() uwtable(sync) {
  ret void
}
