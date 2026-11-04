; RUN: rm -rf %t && split-file %s %t
; RUN: llvm-link %t/decl.ll %t/def.ll -S | FileCheck %s

; Tests that when linking a declaration module (without !callgraph metadata for unprototyped decl)
; and a definition module (with full reconstructed !callgraph metadata),
; the merged definition replaces the declaration and retains the definition's !callgraph metadata.

; CHECK: define dso_local void @bar()
; CHECK: call void (i32, i32, ...) %0(i32 noundef 1, i32 noundef 2), !callee_type [[F_CT:![0-9]+]]
; CHECK: define dso_local void @foo(i32 noundef %a, i32 noundef %0) !callgraph [[F_DEF:![0-9]+]]

; CHECK: [[F_CT]] = !{[[F_DEF]]}
; CHECK: [[F_DEF]] = !{!"_ZTSFviiE"}

;--- decl.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux"

define dso_local void @bar() {
entry:
  %fp = alloca ptr, align 8
  store ptr @foo, ptr %fp, align 8
  %0 = load ptr, ptr %fp, align 8
  call void (i32, i32, ...) %0(i32 noundef 1, i32 noundef 2), !callee_type !1
  ret void
}

declare void @foo(...)

!1 = !{!2}
!2 = !{!"_ZTSFviiE"}

;--- def.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux"

define dso_local void @foo(i32 noundef %a, i32 noundef %0) !callgraph !1 {
entry:
  ret void
}

!1 = !{!"_ZTSFviiE"}
