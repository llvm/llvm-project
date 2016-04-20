; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Function Attrs: nounwind readnone
declare i8* @llvm.thread.pointer() #1

define i8* @thread_pointer() {
; CHECK: thread_pointer:
; CHECK: ear [[REG1:%r[0-5]]], %a0
; CHECK: sllg %r2, [[REG1]], 32
; CHECK: ear %r2, %a1
; CHECK: br %r14
  %1 = tail call i8* @llvm.thread.pointer()
  ret i8* %1
}
