; RUN: llc -o - -mtriple=nvptx64 %s | FileCheck %s

; CHECK-LABEL: .func breakpoint
define void @breakpoint() {
  ; CHECK: brkpt;
  call void @llvm.debugtrap()
  ret void
}
