; RUN: llc -o - -O0 < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @test1(i32 %x) #0 {
entry:
  %tobool = icmp ne i32 %x, 0
  call void @callee1(i1 zeroext %tobool)
  ret void
}

; CHECK-LABEL: test1:
; CHECK:      cmpl   $0, %edi
; CHECK-NEXT: setne  %al
; CHECK-NEXT: andb   $1, %al
; CHECK-NEXT: movzbl %al, %edi
; CHECK-NEXT: callq  callee1

define void @test2(i32 %x) #0 {
entry:
  %tobool = icmp ne i32 %x, 0
  call void @callee2(i1 signext %tobool)
  ret void
}

; CHECK-LABEL: test2:
; CHECK:      cmpl   $0, %edi
; CHECK-NEXT: setne  %al
; CHECK-NEXT: kmovb  %eax, %k0
; CHECK-NEXT: kmovw  %k0, %edi
; CHECK-NEXT: andl  $1, %edi
; CHECK-NEXT: movb  %dil, %al
; CHECK-NEXT: xorl  %edi, %edi
; CHECK-NEXT: testb  %al, %al
; CHECK-NEXT: movl  $-1, %ecx
; CHECK-NEXT: cmovnel  %ecx, %edi
; CHECK-NEXT: callq  callee2

declare void @callee1(i1 zeroext)
declare void @callee2(i1 signext)

attributes #0 = { nounwind "target-cpu"="skylake-avx512" }
