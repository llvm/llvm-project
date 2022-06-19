; RUN: opt < %s -passes='asan-pipeline' -S -o %t.ll
; RUN: FileCheck %s < %t.ll

; Don't do stack malloc on functions containing inline assembly on 64-bit
; platforms. It makes LLVM run out of registers.

; CHECK-LABEL: define void @TestAbsenceOfStackMalloc(i8* %S, i32 %pS, i8* %D, i32 %pD, i32 %h)
; CHECK: %MyAlloca
; CHECK-NOT: call {{.*}} @__asan_stack_malloc

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

define void @TestAbsenceOfStackMalloc(i8* %S, i32 %pS, i8* %D, i32 %pD, i32 %h) #0 {
entry:
  %S.addr = alloca i8*, align 8
  %pS.addr = alloca i32, align 4
  %D.addr = alloca i8*, align 8
  %pD.addr = alloca i32, align 4
  %h.addr = alloca i32, align 4
  %sr = alloca i32, align 4
  %pDiffD = alloca i32, align 4
  %pDiffS = alloca i32, align 4
  %flagSA = alloca i8, align 1
  %flagDA = alloca i8, align 1
  store i8* %S, i8** %S.addr, align 8
  store i32 %pS, i32* %pS.addr, align 4
  store i8* %D, i8** %D.addr, align 8
  store i32 %pD, i32* %pD.addr, align 4
  store i32 %h, i32* %h.addr, align 4
  store i32 4, i32* %sr, align 4
  %0 = load i32, i32* %pD.addr, align 4
  %sub = sub i32 %0, 5
  store i32 %sub, i32* %pDiffD, align 4
  %1 = load i32, i32* %pS.addr, align 4
  %shl = shl i32 %1, 1
  %sub1 = sub i32 %shl, 5
  store i32 %sub1, i32* %pDiffS, align 4
  %2 = load i32, i32* %pS.addr, align 4
  %and = and i32 %2, 15
  %cmp = icmp eq i32 %and, 0
  %conv = zext i1 %cmp to i32
  %conv2 = trunc i32 %conv to i8
  store i8 %conv2, i8* %flagSA, align 1
  %3 = load i32, i32* %pD.addr, align 4
  %and3 = and i32 %3, 15
  %cmp4 = icmp eq i32 %and3, 0
  %conv5 = zext i1 %cmp4 to i32
  %conv6 = trunc i32 %conv5 to i8
  store i8 %conv6, i8* %flagDA, align 1
  call void asm sideeffect "mov\09\09\09$0,\09\09\09\09\09\09\09\09\09\09%rsi\0Amov\09\09\09$2,\09\09\09\09\09\09\09\09\09\09%rcx\0Amov\09\09\09$1,\09\09\09\09\09\09\09\09\09\09%rdi\0Amov\09\09\09$8,\09\09\09\09\09\09\09\09\09\09%rax\0A", "*m,*m,*m,*m,*m,*m,*m,*m,*m,~{rsi},~{rdi},~{rax},~{rcx},~{rdx},~{memory},~{dirflag},~{fpsr},~{flags}"(i8** elementtype(i8*) %S.addr, i8** elementtype(i8*) %D.addr, i32* elementtype(i32) %pS.addr, i32* elementtype(i32) %pDiffS, i32* elementtype(i32) %pDiffD, i32* elementtype(i32) %sr, i8* elementtype(i8) %flagSA, i8* elementtype(i8) %flagDA, i32* elementtype(i32) %h.addr) #1
  ret void
}

attributes #0 = { nounwind sanitize_address }
attributes #1 = { nounwind }
