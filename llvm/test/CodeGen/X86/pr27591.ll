; RUN: llc -o - -O0 < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @test1(i32 %x) #0 {
; CHECK-LABEL: test1:
; CHECK:       # BB#0: # %entry
; CHECK-NEXT:    pushq %rax
; CHECK-NEXT:    testl %edi, %edi
; CHECK-NEXT:    setne %al
; CHECK-NEXT:    movb %al, %cl
; CHECK-NEXT:    kmovw %ecx, %k0
; CHECK-NEXT:    kmovb %k0, %eax
; CHECK-NEXT:    andb $1, %al
; CHECK-NEXT:    movzbl %al, %edi
; CHECK-NEXT:    callq callee1
; CHECK-NEXT:    popq %rax
; CHECK-NEXT:    retq
entry:
  %tobool = icmp ne i32 %x, 0
  call void @callee1(i1 zeroext %tobool)
  ret void
}

define void @test2(i32 %x) #0 {
; CHECK-LABEL: test2:
; CHECK:       # BB#0: # %entry
; CHECK-NEXT:    pushq %rax
; CHECK-NEXT:    testl %edi, %edi
; CHECK-NEXT:    setne %al
; CHECK-NEXT:    movb %al, %cl
; CHECK-NEXT:    kmovw %ecx, %k0
; CHECK-NEXT:    kmovw %k0, %ecx
; CHECK-NEXT:    movb %cl, %al
; CHECK-NEXT:    xorl %edi, %edi
; CHECK-NEXT:    testb %al, %al
; CHECK-NEXT:    movl $-1, %edx
; CHECK-NEXT:    cmovnel %edx, %edi
; CHECK-NEXT:    callq callee2
; CHECK-NEXT:    popq %rax
; CHECK-NEXT:    retq
entry:
  %tobool = icmp ne i32 %x, 0
  call void @callee2(i1 signext %tobool)
  ret void
}

declare void @callee1(i1 zeroext)
declare void @callee2(i1 signext)

attributes #0 = { nounwind "target-cpu"="skylake-avx512" }
