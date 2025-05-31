; RUN: llc < %s -mtriple=sparc-unknown-linux-gnu | FileCheck %s
; RUN: llc < %s -mtriple=sparc64-unknown-linux-gnu | FileCheck %s

; Function Attrs: nounwind readnone
declare ptr @llvm.thread.pointer() #1

define ptr @thread_pointer() {
; CHECK: mov %g7, %o0
  %1 = tail call ptr @llvm.thread.pointer()
  ret ptr %1
}
