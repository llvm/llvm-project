; RUN: llc -mtriple arm-linux-gnueabi -o - %s | FileCheck %s -check-prefix=CHECK-SOFT
; RUN: llc -mtriple arm-linux-gnueabi -mattr=+read-tp-hard -o - %s | FileCheck %s -check-prefix=CHECK-HARD
; RUN: llc -mtriple thumbv7-linux-gnueabi -o - %s | FileCheck %s -check-prefix=CHECK-SOFT
; RUN: llc -mtriple thumbv7-linux-gnueabi -mattr=+read-tp-hard -o - %s | FileCheck %s -check-prefix=CHECK-HARD

declare ptr @llvm.thread.pointer()

define ptr @test() {
entry:
  %tmp1 = call ptr @llvm.thread.pointer()
  ret ptr %tmp1
}

; CHECK-SOFT: bl __aeabi_read_tp
; CHECK-HARD: mrc p15, #0, {{r[0-9]+}}, c13, c0, #3

