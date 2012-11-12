; RUN: llc < %s -mtriple=x86_64-apple-darwin10 -mcpu=corei7 | FileCheck %s

define i64 @test1(i64 %x) nounwind {
entry:
  %cmp = icmp eq i64 %x, 2
  %add = add i64 %x, 1
  %retval.0 = select i1 %cmp, i64 2, i64 %add
  ret i64 %retval.0

; CHECK: test1:
; CHECK: leaq 1(%rdi), %rax
; CHECK: cmpq $2, %rdi
; CHECK: cmoveq %rdi, %rax
; CHECK: ret

}
