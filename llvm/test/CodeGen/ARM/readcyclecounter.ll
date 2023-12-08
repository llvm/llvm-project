; RUN: llc -mtriple=armv7-none-linux-gnueabi < %s | FileCheck %s
; RUN: llc -mtriple=thumbv7-none-linux-gnueabi < %s | FileCheck %s
; RUN: llc -mtriple=armv7-none-linux-gnueabi -mattr=-perfmon < %s | FileCheck %s --check-prefix=CHECK-NO-PERFMON
; RUN: llc -mtriple=armv7m-none-linux-gnueabi < %s | FileCheck %s --check-prefix=CHECK-NO-PERFMON
; RUN: llc -mtriple=armv6-none-linux-gnueabi < %s | FileCheck %s --check-prefix=CHECK-NO-PERFMON

; The performance monitor we're looking for is an ARMv7 extension. It should be
; possible to disable it, but realistically present on at least every v7-A
; processor (but not on v6, at least by default).

declare i64 @llvm.readcyclecounter()

define i64 @get_count() {
  %val = call i64 @llvm.readcyclecounter()
  ret i64 %val

  ; As usual, exact registers only sort of matter but the cycle-count had better
  ; end up in r0 in the end.

; CHECK: mrc p15, #0, r0, c9, c13, #0
; CHECK: {{movs?}} r1, #0

; CHECK-NO-PERFMON: {{movs?}} r0, #0
; CHECK-NO-PERFMON: {{movs?}} r1, #0
}
