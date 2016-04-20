; RUN: llc < %s -march=aarch64 -mtriple=aarch64-linux-gnu | FileCheck %s

; Function Attrs: nounwind readnone
declare i8* @llvm.thread.pointer() #1

define i8* @thread_pointer() {
; CHECK: thread_pointer:
; CHECK: mrs {{x[0-9]+}}, TPIDR_EL0
  %1 = tail call i8* @llvm.thread.pointer()
  ret i8* %1
}
