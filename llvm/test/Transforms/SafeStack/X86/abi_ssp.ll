; RUN: opt -safe-stack -S -mtriple=i686-pc-linux-gnu < %s -o - | FileCheck --check-prefixes=COMMON,TLS32 %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck --check-prefixes=COMMON,TLS64 %s

; RUN: opt -safe-stack -S -mtriple=i686-linux-android < %s -o - | FileCheck --check-prefixes=COMMON,GLOBAL32 %s
; RUN: opt -safe-stack -S -mtriple=i686-linux-android24 < %s -o - | FileCheck --check-prefixes=COMMON,TLS32 %s

; RUN: opt -safe-stack -S -mtriple=x86_64-linux-android < %s -o - | FileCheck --check-prefixes=COMMON,TLS64 %s

define void @foo() safestack sspreq {
entry:
; TLS32: %[[StackGuard:.*]] = load i8*, i8* addrspace(256)* inttoptr (i32 20 to i8* addrspace(256)*)
; TLS64: %[[StackGuard:.*]] = load i8*, i8* addrspace(257)* inttoptr (i32 40 to i8* addrspace(257)*)
; GLOBAL32: %[[StackGuard:.*]] = load i8*, i8** @__stack_chk_guard
; COMMON:   store i8* %[[StackGuard]], i8** %[[StackGuardSlot:.*]]
  %a = alloca i8, align 1
  call void @Capture(i8* %a)

; COMMON: %[[A:.*]] = load i8*, i8** %[[StackGuardSlot]]
; COMMON: icmp ne i8* %[[StackGuard]], %[[A]]
  ret void
}

declare void @Capture(i8*)
