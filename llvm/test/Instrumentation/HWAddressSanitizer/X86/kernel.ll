; Test kernel hwasan instrumentation.
; Generic code is covered by ../kernel.ll, only the x86_64 specific code is
; tested here.
;
; RUN: opt < %s -hwasan -hwasan-kernel=1 -S | FileCheck %s --allow-empty --check-prefixes=INIT
; RUN: opt < %s -hwasan -hwasan-kernel=1 -S | FileCheck %s
; RUN: opt < %s -hwasan -hwasan-kernel=1 -hwasan-mapping-offset=12345678 -S | FileCheck %s
; RUN: opt < %s -hwasan -hwasan-kernel=1 -hwasan-recover=0 -S | FileCheck %s --check-prefixes=CHECK,ABORT
; RUN: opt < %s -hwasan -hwasan-kernel=1 -hwasan-recover=1 -S | FileCheck %s --check-prefixes=CHECK,RECOVER

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i8 @test_load(i8* %a) sanitize_hwaddress {
; CHECK-LABEL: @test_load(
; CHECK: %[[A:[^ ]*]] = ptrtoint i8* %a to i64

; ABORT: call void asm sideeffect "int3\0Anopl 64(%rax)", "{rdi}"(i64 %[[A]])
; ABORT: unreachable
; RECOVER: call void asm sideeffect "int3\0Anopl 96(%rax)", "{rdi}"(i64 %[[A]])
; RECOVER: br label

; CHECK: %[[A:[^ ]*]] = ptrtoint i8* %a to i64
; CHECK: %[[UNTAGGED:[^ ]*]] = or i64 %[[A]], -72057594037927936
; CHECK: %[[UNTAGGED_PTR:[^ ]*]] = inttoptr i64 %[[UNTAGGED]] to i8*
; CHECK: %[[G:[^ ]*]] = load i8, i8* %[[UNTAGGED_PTR]], align 4
; CHECK: ret i8 %[[G]]

entry:
  %b = load i8, i8* %a, align 4
  ret i8 %b
}

; INIT-NOT: call void @__hwasan_init
