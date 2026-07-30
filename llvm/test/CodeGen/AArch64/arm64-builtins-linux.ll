; RUN: llc < %s -mtriple=aarch64-linux-gnu | FileCheck %s
; RUN: llc < %s -mtriple=aarch64-fuchsia | FileCheck %s
; RUN: llc < %s -mtriple=aarch64-linux-gnu -mattr=+tpidrro-el0 | FileCheck --check-prefix=USEROEL0 %s
; RUN: llc < %s -mtriple=aarch64-linux-gnu -mattr=+tpidr-el1 | FileCheck --check-prefix=USEEL1 %s
; RUN: llc < %s -mtriple=aarch64-linux-gnu -mattr=+tpidr-el2 | FileCheck --check-prefix=USEEL2 %s
; RUN: llc < %s -mtriple=aarch64-linux-gnu -mattr=+tpidr-el3 | FileCheck --check-prefix=USEEL3 %s

; Function Attrs: nounwind readnone
declare ptr @llvm.thread.pointer() #1

define ptr @thread_pointer() {
; CHECK: thread_pointer:
; CHECK: mrs {{x[0-9]+}}, TPIDR_EL0
; USEROEL0: thread_pointer:
; USEROEL0: mrs {{x[0-9]+}}, TPIDRRO_EL0
; USEEL1: thread_pointer:
; USEEL1: mrs {{x[0-9]+}}, TPIDR_EL1
; USEEL2: thread_pointer:
; USEEL2: mrs {{x[0-9]+}}, TPIDR_EL2
; USEEL3: thread_pointer:
; USEEL3: mrs {{x[0-9]+}}, TPIDR_EL3
  %1 = tail call ptr @llvm.thread.pointer()
  ret ptr %1
}
