; This test checks that we are not instrumenting unnecessary globals
; RUN: opt < %s -passes=asan -S | FileCheck %s

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

@v_available_externally = available_externally global i32 zeroinitializer
; CHECK-NOT: {{asan_gen.*v_available_externally}}

; CHECK: @asan.module_ctor
