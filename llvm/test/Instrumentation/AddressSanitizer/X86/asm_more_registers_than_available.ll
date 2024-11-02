; RUN: opt < %s -passes=asan -S -o %t.ll
; RUN: FileCheck %s < %t.ll

; Don't do stack malloc on functions containing inline assembly on 64-bit
; platforms. It makes LLVM run out of registers.

; CHECK-LABEL: define void @TestAbsenceOfStackMalloc(ptr %S, i32 %pS, ptr %D, i32 %pD, i32 %h)
; CHECK: %MyAlloca
; CHECK-NOT: call {{.*}} @__asan_stack_malloc

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

define void @TestAbsenceOfStackMalloc(ptr %S, i32 %pS, ptr %D, i32 %pD, i32 %h) #0 {
entry:
  %S.addr = alloca ptr, align 8
  %pS.addr = alloca i32, align 4
  %D.addr = alloca ptr, align 8
  %pD.addr = alloca i32, align 4
  %h.addr = alloca i32, align 4
  %sr = alloca i32, align 4
  %pDiffD = alloca i32, align 4
  %pDiffS = alloca i32, align 4
  %flagSA = alloca i8, align 1
  %flagDA = alloca i8, align 1
  store ptr %S, ptr %S.addr, align 8
  store i32 %pS, ptr %pS.addr, align 4
  store ptr %D, ptr %D.addr, align 8
  store i32 %pD, ptr %pD.addr, align 4
  store i32 %h, ptr %h.addr, align 4
  store i32 4, ptr %sr, align 4
  %0 = load i32, ptr %pD.addr, align 4
  %sub = sub i32 %0, 5
  store i32 %sub, ptr %pDiffD, align 4
  %1 = load i32, ptr %pS.addr, align 4
  %shl = shl i32 %1, 1
  %sub1 = sub i32 %shl, 5
  store i32 %sub1, ptr %pDiffS, align 4
  %2 = load i32, ptr %pS.addr, align 4
  %and = and i32 %2, 15
  %cmp = icmp eq i32 %and, 0
  %conv = zext i1 %cmp to i32
  %conv2 = trunc i32 %conv to i8
  store i8 %conv2, ptr %flagSA, align 1
  %3 = load i32, ptr %pD.addr, align 4
  %and3 = and i32 %3, 15
  %cmp4 = icmp eq i32 %and3, 0
  %conv5 = zext i1 %cmp4 to i32
  %conv6 = trunc i32 %conv5 to i8
  store i8 %conv6, ptr %flagDA, align 1
  call void asm sideeffect "mov\09\09\09$0,\09\09\09\09\09\09\09\09\09\09%rsi\0Amov\09\09\09$2,\09\09\09\09\09\09\09\09\09\09%rcx\0Amov\09\09\09$1,\09\09\09\09\09\09\09\09\09\09%rdi\0Amov\09\09\09$8,\09\09\09\09\09\09\09\09\09\09%rax\0A", "*m,*m,*m,*m,*m,*m,*m,*m,*m,~{rsi},~{rdi},~{rax},~{rcx},~{rdx},~{memory},~{dirflag},~{fpsr},~{flags}"(ptr elementtype(ptr) %S.addr, ptr elementtype(ptr) %D.addr, ptr elementtype(i32) %pS.addr, ptr elementtype(i32) %pDiffS, ptr elementtype(i32) %pDiffD, ptr elementtype(i32) %sr, ptr elementtype(i8) %flagSA, ptr elementtype(i8) %flagDA, ptr elementtype(i32) %h.addr) #1
  ret void
}

attributes #0 = { nounwind sanitize_address }
attributes #1 = { nounwind }
