; RUN: llc < %s -march=x86-64 -no-integrated-as -mtriple=x86_64-linux-gnu | FileCheck %s

; CHECK-LABEL: test1:
; CHECK: movl	(%rdi), %eax
; CHECK: nop
; CHECK: movl	%eax, (%rdi)
; CHECK: ret
define void @test1(i32* %l) {
  %load = load i32, i32* %l
  call void asm "nop", "=*rmrm,0m0m,~{dirflag},~{fpsr},~{flags}"(i32* %l, i32 %load)
  ret void
}
