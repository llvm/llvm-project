; RUN: llc -mtriple arm-linux-gnueabi -o - %s | FileCheck %s -check-prefix=CHECK-SOFT
; RUN: llc -mtriple arm-linux-gnueabi -mattr=+read-tp-tpidrurw -o - %s | FileCheck %s -check-prefix=CHECK-TPIDRURW
; RUN: llc -mtriple arm-linux-gnueabi -mattr=+read-tp-tpidruro -o - %s | FileCheck %s -check-prefix=CHECK-TPIDRURO
; RUN: llc -mtriple arm-linux-gnueabi -mattr=+read-tp-tpidrprw -o - %s | FileCheck %s -check-prefix=CHECK-TPIDRPRW
; RUN: llc -mtriple thumbv7-linux-gnueabi -o - %s | FileCheck %s -check-prefix=CHECK-SOFT
; RUN: llc -mtriple thumbv7-linux-gnueabi -mattr=+read-tp-tpidrurw -o - %s | FileCheck %s -check-prefix=CHECK-TPIDRURW
; RUN: llc -mtriple thumbv7-linux-gnueabi -mattr=+read-tp-tpidruro -o - %s | FileCheck %s -check-prefix=CHECK-TPIDRURO
; RUN: llc -mtriple thumbv7-linux-gnueabi -mattr=+read-tp-tpidrprw -o - %s | FileCheck %s -check-prefix=CHECK-TPIDRPRW

declare ptr @llvm.thread.pointer()

define ptr @test() {
entry:
  %tmp1 = call ptr @llvm.thread.pointer()
  ret ptr %tmp1
}

; CHECK-SOFT:     bl __aeabi_read_tp
; CHECK-TPIDRURW: mrc p15, #0, {{r[0-9]+}}, c13, c0, #2
; CHECK-TPIDRURO: mrc p15, #0, {{r[0-9]+}}, c13, c0, #3
; CHECK-TPIDRPRW: mrc p15, #0, {{r[0-9]+}}, c13, c0, #4

