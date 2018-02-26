; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -mattr=+sse2 -asm-instrumentation=address -asan-instrument-assembly | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: rep_movs_1b
; CHECK: pushfq
; CHECK-NEXT: testq %rcx, %rcx
; CHECK-NEXT: je [[B:.*]]

; CHECK: leaq -128(%rsp), %rsp
; CHECK-NEXT: pushq %rax
; CHECK-NEXT: pushq %rdx
; CHECK-NEXT: pushq %rbx
; CHECK-NEXT: pushfq

; CHECK: leaq (%rsi), %rdx
; CHECK: movq %rdx, %rdi
; CHECK-NEXT: callq __asan_report_load1@PLT

; CHECK: leaq -1(%rsi,%rcx), %rdx
; CHECK: movq %rdx, %rdi
; CHECK-NEXT: callq __asan_report_load1@PLT

; CHECK: leaq (%rdi), %rdx
; CHECK: movq %rdx, %rdi
; CHECK-NEXT: callq __asan_report_store1@PLT

; CHECK: leaq -1(%rdi,%rcx), %rdx
; CHECK: movq %rdx, %rdi
; CHECK-NEXT: callq __asan_report_store1@PLT

; CHECK: popfq
; CHECK-NEXT: popq %rbx
; CHECK-NEXT: popq %rdx
; CHECK-NEXT: popq %rax
; CHECK-NEXT: leaq 128(%rsp), %rsp

; CHECK: [[B]]:
; CHECK-NEXT: popfq

; CHECK: rep movsb (%rsi), %es:(%rdi)

; Function Attrs: nounwind sanitize_address uwtable
define void @rep_movs_1b(i8* %dst, i8* %src, i64 %n) #0 {
entry:
  tail call void asm sideeffect "rep movsb \0A\09", "{si},{di},{cx},~{memory},~{dirflag},~{fpsr},~{flags}"(i8* %src, i8* %dst, i64 %n) #1
  ret void
}

; CHECK-LABEL: rep_movs_8b
; CHECK: pushfq
; CHECK-NEXT: testq %rcx, %rcx
; CHECK-NEXT: je [[Q:.*]]

; CHECK: leaq (%rsi), %rdx
; CHECK: movq %rdx, %rdi
; CHECK-NEXT: callq __asan_report_load8@PLT

; CHECK: leaq -1(%rsi,%rcx,8), %rdx
; CHECK: movq %rdx, %rdi
; CHECK-NEXT: callq __asan_report_load8@PLT

; CHECK: leaq (%rdi), %rdx
; CHECK: movq %rdx, %rdi
; CHECK-NEXT: callq __asan_report_store8@PLT

; CHECK: leaq -1(%rdi,%rcx,8), %rdx
; CHECK: movq %rdx, %rdi
; CHECK-NEXT: callq __asan_report_store8@PLT

; CHECK: [[Q]]:
; CHECK-NEXT: popfq

; CHECK: rep movsq (%rsi), %es:(%rdi)

; Function Attrs: nounwind sanitize_address uwtable
define void @rep_movs_8b(i64* %dst, i64* %src, i64 %n) #0 {
entry:
  tail call void asm sideeffect "rep movsq \0A\09", "{si},{di},{cx},~{memory},~{dirflag},~{fpsr},~{flags}"(i64* %src, i64* %dst, i64 %n) #1
  ret void
}

attributes #0 = { nounwind sanitize_address uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
