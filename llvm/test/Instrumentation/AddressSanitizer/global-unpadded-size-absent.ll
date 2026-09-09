; Globals the pass declines to instrument keep no size metadata, so consumers
; can treat its absence as "the allocated size is the declared size".

; RUN: opt < %s -passes=asan -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Alignment exceeds the minimum redzone size.
@overaligned = global i32 0, align 64

@tls = thread_local global i32 0, align 4

@declared = external global i32, align 4

@opted_out = global i32 0, align 4, no_sanitize_address

; CHECK-NOT: sanitize.unpadded.size
