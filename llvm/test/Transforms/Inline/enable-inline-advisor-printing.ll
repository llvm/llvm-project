; RUN: opt -passes=inliner-wrapper -keep-inline-advisor-for-printing \
; RUN:     -enable-scc-inline-advisor-printing -S < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local noundef i32 @_Z3fooi(i32 noundef %y) {
entry:
  ret i32 %y
}

define dso_local noundef i32 @main(i32 noundef %argc, ptr noundef %argv) {
entry:
  %call = call noundef i32 @_Z3fooi(i32 noundef %argc)
  ret i32 %call
}

; CHECK-COUNT-4: Unimplemented InlineAdvisor print
