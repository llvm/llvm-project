target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

; RUN: opt -S -passes=globalopt < %s | FileCheck %s

; Verify that the initialization of the available_externally global is not eliminated
; CHECK: @llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @foo_static_init, i8* null }]

@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @foo_static_init, i8* null }]
@foo_external = available_externally global void ()* null

define internal void @foo_static_init() {
entry:
  store void ()* @foo_impl, void ()** @foo_external
  ret void
}

define internal void @foo_impl() {
entry:
  ret void
}

