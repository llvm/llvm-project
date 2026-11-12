; RUN: llc < %s -mtriple=armv7 | FileCheck %s

declare ptr @llvm.stackaddress.p0()

define ptr @test() {
; CHECK: mov r0, sp
; CHECK: bx  lr
  %sp = call ptr @llvm.stackaddress.p0()
  ret ptr %sp
}
