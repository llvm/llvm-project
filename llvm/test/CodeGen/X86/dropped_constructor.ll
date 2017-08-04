; Test to ensure that a global value that was dropped to a declaration
; (e.g. ThinLTO will drop non-prevailing weak to declarations) does not
; provoke creation of a comdat when it had an initializer.
; RUN: llc -mtriple x86_64-unknown-linux-gnu < %s | FileCheck %s
; CHECK-NOT: comdat

; ModuleID = 'dropped_constructor.o'
source_filename = "dropped_constructor.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@fv = external global i8, align 8
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @__cxx_global_var_init.33, i8* @fv }]

; Function Attrs: norecurse nounwind
define internal void @__cxx_global_var_init.33() section ".text.startup" {
  store i8 1, i8* @fv, align 8
  ret void
}
