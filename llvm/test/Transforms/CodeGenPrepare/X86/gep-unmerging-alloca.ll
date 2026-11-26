; RUN: opt -passes='require<profile-summary>,function(codegenprepare)' -S -mtriple=x86_64 < %s | FileCheck %s
; RUN: llc -O2 -mtriple=x86_64 -verify-machineinstrs < %s -o /dev/null

; Unmerging the GEPs based on a static alloca must not cycle with address
; sinking, which considers the alloca to be live throughout the function.
define ptr @static_alloca(ptr %target) {
; CHECK-LABEL: define ptr @static_alloca(
; CHECK: %base = alloca [32 x i8], align 8
; CHECK: indirectbr ptr %target
; CHECK: %p24 = getelementptr i8, ptr %base, i64 24
; CHECK-NEXT: ret ptr %p24
entry:
  %base = alloca [32 x i8], align 8
  %p8 = getelementptr i8, ptr %base, i64 8
  indirectbr ptr %target, [label %dest]
dest:
  %v8 = load i64, ptr %p8, align 8
  %p16 = getelementptr i8, ptr %base, i64 16
  %v16 = load ptr, ptr %p16, align 8
  %p24 = getelementptr i8, ptr %base, i64 24
  ret ptr %p24
}

; A loaded base can still benefit from unmerging: only %p8, not %base, needs
; to remain live across the indirect branch.
define ptr @loaded_base(ptr %target, ptr %slot) {
; CHECK-LABEL: define ptr @loaded_base(
; CHECK: %base = load ptr, ptr %slot, align 8
; CHECK: %p8 = getelementptr i8, ptr %base, i64 8
; CHECK: indirectbr ptr %target
; CHECK: %p24 = getelementptr i8, ptr %p8, i64 16
; CHECK-NEXT: ret ptr %p24
entry:
  %base = load ptr, ptr %slot, align 8
  %p8 = getelementptr i8, ptr %base, i64 8
  indirectbr ptr %target, [label %dest]
dest:
  %v8 = load i64, ptr %p8, align 8
  %p16 = getelementptr i8, ptr %base, i64 16
  %v16 = load ptr, ptr %p16, align 8
  %p24 = getelementptr i8, ptr %base, i64 24
  ret ptr %p24
}

; A dynamic alloca is not a fixed frame reference and remains eligible.
define ptr @dynamic_alloca(ptr %target, i64 %size) {
; CHECK-LABEL: define ptr @dynamic_alloca(
; CHECK: %base = alloca i8, i64 %size, align 8
; CHECK: %p8 = getelementptr i8, ptr %base, i64 8
; CHECK: indirectbr ptr %target
; CHECK: %p24 = getelementptr i8, ptr %p8, i64 16
; CHECK-NEXT: ret ptr %p24
entry:
  %base = alloca i8, i64 %size, align 8
  %p8 = getelementptr i8, ptr %base, i64 8
  indirectbr ptr %target, [label %dest]
dest:
  %v8 = load i64, ptr %p8, align 8
  %p16 = getelementptr i8, ptr %base, i64 16
  %v16 = load ptr, ptr %p16, align 8
  %p24 = getelementptr i8, ptr %base, i64 24
  ret ptr %p24
}
