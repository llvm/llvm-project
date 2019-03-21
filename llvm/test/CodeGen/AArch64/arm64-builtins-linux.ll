; RUN: llc < %s -mtriple=aarch64-linux-gnu | FileCheck %s
; RUN: llc < %s -mtriple=aarch64-fuchsia | FileCheck %s
; RUN: llc < %s -mtriple=aarch64-fuchsia -code-model=kernel | FileCheck --check-prefix=FUCHSIA-KERNEL %s
; RUN: llc < %s -mtriple=aarch64-linux-gnu -mattr=+tpidr-el1 | FileCheck --check-prefix=USEEL1 %s
; RUN: llc < %s -mtriple=aarch64-linux-gnu -mattr=+tpidr-el2 | FileCheck --check-prefix=USEEL2 %s
; RUN: llc < %s -mtriple=aarch64-linux-gnu -mattr=+tpidr-el3 | FileCheck --check-prefix=USEEL3 %s

; Function Attrs: nounwind readnone
declare i8* @llvm.thread.pointer() #1

define i8* @thread_pointer() {
; CHECK: thread_pointer:
; CHECK: mrs {{x[0-9]+}}, TPIDR_EL0
; FUCHSIA-KERNEL: thread_pointer:
; FUCHSIA-KERNEL: mrs {{x[0-9]+}}, TPIDR_EL1
; USEEL1: thread_pointer:
; USEEL1: mrs {{x[0-9]+}}, TPIDR_EL1
; USEEL2: thread_pointer:
; USEEL2: mrs {{x[0-9]+}}, TPIDR_EL2
; USEEL3: thread_pointer:
; USEEL3: mrs {{x[0-9]+}}, TPIDR_EL3
  %1 = tail call i8* @llvm.thread.pointer()
  ret i8* %1
}
