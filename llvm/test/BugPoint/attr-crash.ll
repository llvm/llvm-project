; RUN: bugpoint -load %llvmshlibdir/BugpointPasses%shlibext %s -output-prefix %t -bugpoint-crashfuncattr 2>&1 | FileCheck %s
; REQUIRES: plugins
;
; ModuleID = 'attr-crash.ll'
source_filename = "test.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main(i32 %argc, i8** %argv) local_unnamed_addr #0 {
  ret i32 0
}

; CHECK-NOT: Attribute 'optnone' requires 'noinline'!
attributes #0 = { noinline nounwind optnone uwtable "bugpoint-crash" }
