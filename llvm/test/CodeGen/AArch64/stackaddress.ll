; RUN: llc < %s -mtriple=aarch64 | FileCheck %s

declare ptr @llvm.stackaddress.p0()

define ptr @test() {
; CHECK: mov x0, sp
; CHECK: ret
  %sp = call ptr @llvm.stackaddress.p0()
  ret ptr %sp
}
