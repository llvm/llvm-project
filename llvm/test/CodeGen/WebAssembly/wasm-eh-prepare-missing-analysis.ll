; RUN: not opt -passes=wasm-eh-prepare -disable-output %s 2>&1 | FileCheck %s

; CHECK: 'RuntimeLibraryAnalysis' analysis required

target triple = "wasm32-unknown-unknown"

define void @empty() {
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"exception-model", !"wasm"}
