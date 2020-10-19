; RUN: opt -S -hotcoldsplit -hotcoldsplit-threshold=0 < %s | FileCheck %s

; Source:
;
; extern void sideeffect(int);
; extern void __attribute__((cold)) sink();
; void foo(int cond) {
;   if (cond) { //< Start outlining here.
;     while (cond > 10) {
;       --cond;
;       sideeffect(0);
;     }
;     sink();
;   }
;   sideeffect(1);
; }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; CHECK-LABEL: define {{.*}}@foo(
; CHECK: br i1 {{.*}}, label %if.end, label %codeRepl
; CHECK-LABEL: codeRepl:
; CHECK-NEXT: call void @foo.cold.1
; CHECK-LABEL: if.end:
; CHECK: call void @sideeffect(i32 1)
define void @foo(i32 %cond) {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.end, label %while.cond.preheader

while.cond.preheader:                             ; preds = %entry
  %cmp3 = icmp sgt i32 %cond, 10
  br i1 %cmp3, label %while.body.preheader, label %while.end

while.body.preheader:                             ; preds = %while.cond.preheader
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  %cond.addr.04 = phi i32 [ %dec, %while.body ], [ %cond, %while.body.preheader ]
  %dec = add nsw i32 %cond.addr.04, -1
  tail call void @sideeffect(i32 0) #3
  %cmp = icmp sgt i32 %dec, 10
  br i1 %cmp, label %while.body, label %while.end.loopexit

while.end.loopexit:                               ; preds = %while.body
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %while.cond.preheader
  tail call void (...) @sink()
  ret void

if.end:                                           ; preds = %entry
  tail call void @sideeffect(i32 1)
  ret void
}

; This is the same as @foo, but the while loop comes after the sink block.
; CHECK-LABEL: define {{.*}}@while_loop_after_sink(
; CHECK: br i1 {{.*}}, label %if.end, label %codeRepl
; CHECK-LABEL: codeRepl:
; CHECK-NEXT: call void @while_loop_after_sink.cold.1
; CHECK-LABEL: if.end:
; CHECK: call void @sideeffect(i32 1)
define void @while_loop_after_sink(i32 %cond) {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.end, label %sink

sink:
  tail call void (...) @sink()
  br label %while.cond.preheader

while.cond.preheader:
  %cmp3 = icmp sgt i32 %cond, 10
  br i1 %cmp3, label %while.body.preheader, label %while.end

while.body.preheader:                             ; preds = %while.cond.preheader
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  %cond.addr.04 = phi i32 [ %dec, %while.body ], [ %cond, %while.body.preheader ]
  %dec = add nsw i32 %cond.addr.04, -1
  tail call void @sideeffect(i32 0) #3
  %cmp = icmp sgt i32 %dec, 10
  br i1 %cmp, label %while.body, label %while.end.loopexit

while.end.loopexit:                               ; preds = %while.body
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %while.cond.preheader
  ret void

if.end:                                           ; preds = %entry
  tail call void @sideeffect(i32 1)
  ret void
}

; CHECK-LABEL: define {{.*}}@foo.cold.1
; CHECK: phi i32
; CHECK-NEXT: add nsw i32
; CHECK-NEXT: call {{.*}}@sideeffect
; CHECK-NEXT: icmp
; CHECK-NEXT: br

; CHECK-LABEL: define {{.*}}@while_loop_after_sink.cold.1
; CHECK: call {{.*}}@sink
; CHECK: phi i32
; CHECK-NEXT: add nsw i32
; CHECK-NEXT: call {{.*}}@sideeffect
; CHECK-NEXT: icmp
; CHECK-NEXT: br

declare void @sideeffect(i32)

declare void @sink(...) cold
