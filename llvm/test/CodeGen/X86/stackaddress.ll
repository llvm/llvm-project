; RUN: llc < %s -mtriple=x86_64-linux-gnu -o - | FileCheck --check-prefix=x86_64 %s
; RUN: llc < %s -mtriple=i386-linux-gnu -o -   | FileCheck --check-prefix=i386   %s

declare ptr @llvm.stackaddress.p0()

define ptr @test() {
; x86_64: movq %rsp, %rax
; x86_64: retq

; i386: movl %esp, %eax
; i386: retl
  %sp = call ptr @llvm.stackaddress.p0()
  ret ptr %sp
}
