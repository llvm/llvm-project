; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@foo = dso_local global i32 1, align 4

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"SemanticInterposition", float 1.}

; CHECK: SemanticInterposition metadata requires constant integer argument
