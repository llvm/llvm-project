; Test kernel hwasan instrumentation.
;
; RUN: opt < %s -hwasan -hwasan-kernel=1 -S | FileCheck %s --allow-empty --check-prefixes=KERNEL
; RUN: opt < %s -hwasan -hwasan-mapping-offset=12345678 -S | FileCheck %s  --check-prefixes=OFFSET

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android"

define i8 @test_load(i8* %a) sanitize_hwaddress {
; OFFSET-LABEL: @test_load(
; OFFSET: %[[A:[^ ]*]] = ptrtoint i8* %a to i64
; OFFSET: %[[B:[^ ]*]] = lshr i64 %[[A]], 56
; OFFSET: %[[PTRTAG:[^ ]*]] = trunc i64 %[[B]] to i8
; OFFSET: %[[C:[^ ]*]] = and i64 %[[A]], 72057594037927935
; OFFSET: %[[D:[^ ]*]] = lshr i64 %[[C]], 4
; OFFSET: %[[D1:[^ ]*]] = add i64 %[[D]], 12345678
; OFFSET: %[[E:[^ ]*]] = inttoptr i64 %[[D1]] to i8*
; OFFSET: %[[MEMTAG:[^ ]*]] = load i8, i8* %[[E]]
; OFFSET: %[[F:[^ ]*]] = icmp ne i8 %[[PTRTAG]], %[[MEMTAG]]
; OFFSET: br i1 %[[F]],

entry:
  %b = load i8, i8* %a, align 4
  ret i8 %b
}

; KERNEL-NOT: call void @__hwasan_init
