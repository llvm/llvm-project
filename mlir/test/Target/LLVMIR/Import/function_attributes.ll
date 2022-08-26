; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

// -----

; CHECK: llvm.func @readnone_attr() attributes {llvm.readnone}
declare void @readnone_attr() #0
attributes #0 = { readnone }

// -----

; CHECK: llvm.func @readnone_attr() attributes {llvm.readnone} {
; CHECK:   llvm.return
; CHECK: }
define void @readnone_attr() readnone {
entry:
  ret void
}
