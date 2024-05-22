; Check that the unlikely branch is outlined. Override internal branch thresholds with -hotcoldsplit-cold-probability-denom

; RUN: opt -S -passes=hotcoldsplit < %s | FileCheck %s --check-prefixes=CHECK-OUTLINE,CHECK-NOOUTLINE-BAZ
; RUN: opt -S -passes=hotcoldsplit -hotcoldsplit-cold-probability-denom=50 < %s | FileCheck --check-prefixes=CHECK-OUTLINE,CHECK-PROB %s

; int cold(const char*);
; int hot(const char*);
; void foo(int a, int b) {
;   if (a == b) [[unlikely]] { // Should be outlined.
;     cold("same");
;     cold("same");
;   } else {
;     hot("different");
;   }
; }

; void bar(int a, int b) {
;   if (a == b) [[likely]] {
;     hot("same");
;   } else { // Should be outlined.
;     cold("different");
;     cold("different");
;   }
; }

; void baz(int a, int b) {
;   if (a == b) [[likely]] {
;     hot("same");
;   } else { // Should be outlined.
;     cold("different");
;     cold("different");
;   }
; }

; All the outlined cold functions are emitted after the hot functions.
; CHECK-OUTLINE: @foo
; CHECK-OUTLINE: @bar
; CHECK-OUTLINE: @baz

; CHECK-OUTLINE: internal void @foo.cold.1() #[[ATTR0:[0-9]+]]
; CHECK-OUTLINE-NEXT: newFuncRoot
; CHECK-OUTLINE: tail call noundef i32 @cold
; CHECK-OUTLINE: tail call noundef i32 @cold

; CHECK-OUTLINE: internal void @bar.cold.1() #[[ATTR0:[0-9]+]]
; CHECK-OUTLINE-NEXT: newFuncRoot
; CHECK-OUTLINE: tail call noundef i32 @cold
; CHECK-OUTLINE: tail call noundef i32 @cold

; CHECK-NOOUTLINE-BAZ-NOT: internal void @baz.cold.1()

; CHECK-PROB: internal void @baz.cold.1() #[[ATTR0:[0-9]+]]
; CHECK-PROB-NEXT: newFuncRoot
; CHECK-PROB: tail call noundef i32 @cold
; CHECK-PROB: tail call noundef i32 @cold
; CHECK-OUTLINE: attributes #[[ATTR0]] = { cold minsize }

source_filename = "/app/example.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [5 x i8] c"same\00", align 1
@.str.1 = private unnamed_addr constant [10 x i8] c"different\00", align 1

define dso_local void @foo(i32 noundef %a, i32 noundef %b) local_unnamed_addr {
entry:
  %cmp = icmp eq i32 %a, %b
  br i1 %cmp, label %if.then, label %if.else, !prof !1

if.then:
  %call = tail call noundef i32 @cold(ptr noundef nonnull @.str)
  %call1 = tail call noundef i32 @cold(ptr noundef nonnull @.str)
  br label %if.end

if.else:
  %call2 = tail call noundef i32 @hot(ptr noundef nonnull @.str.1)
  br label %if.end

if.end:
  ret void
}

declare noundef i32 @cold(ptr noundef) local_unnamed_addr #1

declare noundef i32 @hot(ptr noundef) local_unnamed_addr #1

define dso_local void @bar(i32 noundef %a, i32 noundef %b) local_unnamed_addr {
entry:
  %cmp = icmp eq i32 %a, %b
  br i1 %cmp, label %if.then, label %if.else, !prof !2

if.then:
  %call = tail call noundef i32 @hot(ptr noundef nonnull @.str)
  br label %if.end

if.else:
  %call1 = tail call noundef i32 @cold(ptr noundef nonnull @.str.1)
  %call2 = tail call noundef i32 @cold(ptr noundef nonnull @.str.1)
  br label %if.end

if.end:
  ret void
}

define dso_local void @baz(i32 noundef %a, i32 noundef %b) local_unnamed_addr {
entry:
  %cmp = icmp eq i32 %a, %b
  br i1 %cmp, label %if.then, label %if.else, !prof !3

if.then:
  %call = tail call noundef i32 @hot(ptr noundef nonnull @.str)
  br label %if.end

if.else:
  %call1 = tail call noundef i32 @cold(ptr noundef nonnull @.str.1)
  %call2 = tail call noundef i32 @cold(ptr noundef nonnull @.str.1)
  br label %if.end

if.end:
  ret void
}

!1 = !{!"branch_weights", i32 1, i32 2000}
!2 = !{!"branch_weights", i32 2000, i32 1}
!3 = !{!"branch_weights", i32 50, i32 1}
