; Test kernel hwasan instrumentation.
;
; RUN: opt < %s -hwasan -hwasan-kernel=1 -S | FileCheck %s --allow-empty --check-prefixes=INIT
; RUN: opt < %s -hwasan -hwasan-kernel=1 -S | FileCheck %s  --check-prefixes=CHECK,NOOFFSET,NO-MATCH-ALL
; RUN: opt < %s -hwasan -hwasan-kernel=1 -hwasan-mapping-offset=12345678 -S | FileCheck %s  --check-prefixes=CHECK,OFFSET,NO-MATCH-ALL
; RUN: opt < %s -hwasan -hwasan-kernel=1 -hwasan-recover=0 -S | FileCheck %s  --check-prefixes=CHECK,NOOFFSET,ABORT,NO-MATCH-ALL
; RUN: opt < %s -hwasan -hwasan-kernel=1 -hwasan-recover=1 -S | FileCheck %s  --check-prefixes=CHECK,NOOFFSET,RECOVER,NO-MATCH-ALL
; RUN: opt < %s -hwasan -hwasan-kernel=1 -hwasan-recover=1 -hwasan-match-all-tag=0xff -S | FileCheck %s  --check-prefixes=CHECK,NOOFFSET,RECOVER,MATCH-ALL

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android"

define i8 @test_load(i8* %a) sanitize_hwaddress {
; CHECK-LABEL: @test_load(
; CHECK: %[[A:[^ ]*]] = ptrtoint i8* %a to i64
; CHECK: %[[B:[^ ]*]] = lshr i64 %[[A]], 56
; CHECK: %[[PTRTAG:[^ ]*]] = trunc i64 %[[B]] to i8
; CHECK: %[[C:[^ ]*]] = or i64 %[[A]], -72057594037927936
; CHECK: %[[D:[^ ]*]] = lshr i64 %[[C]], 4

; NOOFFSET: %[[E:[^ ]*]] = inttoptr i64 %[[D]] to i8*

; OFFSET: %[[D1:[^ ]*]] = add i64 %[[D]], 12345678
; OFFSET: %[[E:[^ ]*]] = inttoptr i64 %[[D1]] to i8*

; CHECK: %[[MEMTAG:[^ ]*]] = load i8, i8* %[[E]]
; CHECK: %[[F:[^ ]*]] = icmp ne i8 %[[PTRTAG]], %[[MEMTAG]]

; MATCH-ALL: %[[G:[^ ]*]] = icmp ne i8 %[[PTRTAG]], -1
; MATCH-ALL: %[[H:[^ ]*]] = and i1 %[[F]], %[[G]]
; MATCH-ALL: br i1 %[[H]], label {{.*}}, label {{.*}}, !prof {{.*}}

; NO-MATCH-ALL: br i1 %[[F]], label {{.*}}, label {{.*}}, !prof {{.*}}

; ABORT: call void asm sideeffect "brk #2304", "{x0}"(i64 %[[A]])
; ABORT: unreachable
; RECOVER: call void asm sideeffect "brk #2336", "{x0}"(i64 %[[A]])
; RECOVER: br label

; CHECK: %[[G:[^ ]*]] = load i8, i8* %a, align 4
; CHECK: ret i8 %[[G]]

entry:
  %b = load i8, i8* %a, align 4
  ret i8 %b
}

; INIT-NOT: call void @__hwasan_init
