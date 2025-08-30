; RUN: llc < %s -mtriple=armv7 | FileCheck %s --check-prefix=armv7
; RUN: llc < %s -mtriple=aarch64 | FileCheck %s --check-prefix=aarch64

declare ptr @llvm.stackaddress.p0()

define ptr @test() {
; armv7: mov r0, sp
; armv7: bx  lr

; aarch64: mov x0, sp
; aarch64: ret
  %sp = call ptr @llvm.stackaddress.p0()
  ret ptr %sp
}
