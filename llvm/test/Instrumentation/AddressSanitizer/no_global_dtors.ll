; Check Default behaviour still emits dtors
; RUN: opt < %s -passes=asan -S | FileCheck -check-prefix=CHECK-DEFAULT %s
; CHECK-DEFAULT: llvm.global_dtor{{.+}}asan.module_dtor
; CHECK-DEFAULT: define internal void @asan.module_dtor

; Check with dtor emission disabled
; RUN: opt < %s -passes=asan -asan-destructor-kind=none -S | FileCheck %s
; CHECK-NOT: llvm.global_dtor{{.+}}asan.module_dtor
; CHECK-NOT: define internal void @asan.module_dtor

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx11.0.0"

@foo = dso_local global i32 0, align 4
