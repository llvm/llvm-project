; RUN: llc < %s -mtriple=sparc   | FileCheck --check-prefix=sparc32 %s
; RUN: llc < %s -mtriple=sparcv9 | FileCheck --check-prefix=sparc64 %s

declare ptr @llvm.stackaddress.p0()

define ptr @test() {
; sparc32: save %sp, -96, %sp
; sparc32: ret
; sparc32: restore %sp, 68, %o0
;
; sparc64: save %sp, -128, %sp
; sparc64: ret
; sparc64: restore %sp, 2175, %o0
  %sp = call ptr @llvm.stackaddress.p0()
  ret ptr %sp
}
