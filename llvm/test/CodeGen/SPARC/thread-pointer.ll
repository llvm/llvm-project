; RUN: llc < %s -mtriple=sparc-unknown-linux-gnu | FileCheck %s
; RUN: llc < %s -mtriple=sparc64-unknown-linux-gnu | FileCheck %s

; Function Attrs: nounwind readnone
declare i8* @llvm.thread.pointer() #1

define i8* @thread_pointer() {
; CHECK: mov %g7, %o0
  %1 = tail call i8* @llvm.thread.pointer()
  ret i8* %1
}
