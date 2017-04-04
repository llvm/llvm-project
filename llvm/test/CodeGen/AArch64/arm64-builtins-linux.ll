; RUN: llc < %s -mtriple=aarch64-linux-gnu | FileCheck %s
; RUN: llc < %s -mtriple=aarch64-fuchsia | FileCheck %s
; RUN: llc < %s -mtriple=aarch64-fuchsia -code-model=kernel | FileCheck --check-prefix=FUCHSIA-KERNEL %s

; Function Attrs: nounwind readnone
declare i8* @llvm.thread.pointer() #1

define i8* @thread_pointer() {
; CHECK: thread_pointer:
; CHECK: mrs {{x[0-9]+}}, TPIDR_EL0
; FUCHSIA-KERNEL: thread_pointer:
; FUCHSIA-KERNEL: mrs {{x[0-9]+}}, TPIDR_EL1
  %1 = tail call i8* @llvm.thread.pointer()
  ret i8* %1
}
