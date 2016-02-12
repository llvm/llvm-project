; MSan converts 2-element global_ctors to 3-element when adding the new entry.
; RUN: opt < %s -msan -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: $msan.module_ctor = comdat any
; CHECK: @llvm.global_ctors = appending global [2 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @f, i8* null }, { i32, void ()*, i8* } { i32 0, void ()* @msan.module_ctor, i8* bitcast (void ()* @msan.module_ctor to i8*) }]

@llvm.global_ctors = appending global [1 x { i32, void ()* }] [{ i32, void ()* } { i32 65535, void ()* @f }]

define internal void @f() {
entry:
  ret void
}

; CHECK: define internal void @msan.module_ctor() comdat {
