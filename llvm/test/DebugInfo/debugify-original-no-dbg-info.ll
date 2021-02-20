; RUN: opt -verify-debuginfo-preserve -instcombine --enable-new-pm=false -S -o - < %s 2>&1 | FileCheck %s

; CHECK: ModuleDebugify (original debuginfo): Skipping module without debug info
; CHECK-NEXT: CheckModuleDebugify (original debuginfo): Skipping module without debug info

; ModuleID = 'no-dbg-info.c'
source_filename = "no-dbg-info.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define dso_local i32 @fn() {
  %1 = call i32 (...) @fn2()
  ret i32 %1
}

declare dso_local i32 @fn2(...)

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 11.0.0"}
