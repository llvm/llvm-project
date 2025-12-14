; RUN: opt -safe-stack -S -mtriple=i686-pc-linux-gnu < %s -o - | FileCheck --check-prefixes=COMMON,TLS32 %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck --check-prefixes=COMMON,TLS64 %s

; RUN: opt -safe-stack -S -mtriple=i686-linux-android < %s -o - | FileCheck --check-prefixes=COMMON,TLS32 %s
; RUN: opt -safe-stack -S -mtriple=x86_64-linux-android < %s -o - | FileCheck --check-prefixes=COMMON,TLS64 %s

; RUN: opt -safe-stack -S -mtriple=x86_64-unknown-fuchsia < %s -o - | FileCheck --check-prefixes=COMMON,FUCHSIA64 %s

; RUN: opt -passes=safe-stack -S -mtriple=i686-pc-linux-gnu < %s -o - | FileCheck --check-prefixes=COMMON,TLS32 %s
; RUN: opt -passes=safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck --check-prefixes=COMMON,TLS64 %s

; RUN: opt -passes=safe-stack -S -mtriple=i686-linux-android < %s -o - | FileCheck --check-prefixes=COMMON,TLS32 %s
; RUN: opt -passes=safe-stack -S -mtriple=x86_64-linux-android < %s -o - | FileCheck --check-prefixes=COMMON,TLS64 %s

; RUN: opt -passes=safe-stack -S -mtriple=x86_64-unknown-fuchsia < %s -o - | FileCheck --check-prefixes=COMMON,FUCHSIA64 %s


define void @foo() safestack sspreq {
entry:
; TLS32: %[[StackGuard:.*]] = load ptr, ptr addrspace(256) inttoptr (i32 20 to ptr addrspace(256))
; TLS64: %[[StackGuard:.*]] = load ptr, ptr addrspace(257) inttoptr (i32 40 to ptr addrspace(257))
; FUCHSIA64: %[[StackGuard:.*]] = load ptr, ptr addrspace(257) inttoptr (i32 16 to ptr addrspace(257))
; GLOBAL32: %[[StackGuard:.*]] = call ptr @llvm.stackguard()
; COMMON:   store ptr %[[StackGuard]], ptr %[[StackGuardSlot:.*]]
  %a = alloca i8, align 1
  call void @Capture(ptr %a)

; COMMON: %[[A:.*]] = load ptr, ptr %[[StackGuardSlot]]
; COMMON: icmp ne ptr %[[StackGuard]], %[[A]]
  ret void
}

declare void @Capture(ptr)
