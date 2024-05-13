; RUN: llc -march=hexagon < %s | FileCheck %s
; REQUIRES: asserts

; Check that we don't crash.
; CHECK: call bar

target triple = "hexagon"

@debug = external hidden unnamed_addr global i1, align 4

; Function Attrs: nounwind
define void @foo(i1 %cond) local_unnamed_addr #0 {
entry:
  br label %if.end5

if.end5:                                          ; preds = %entry
  br i1 undef, label %if.then12, label %if.end13

if.then12:                                        ; preds = %if.end5
  ret void

if.end13:                                         ; preds = %if.end5
  br label %for.cond

for.cond:                                         ; preds = %if.end13
  %or.cond288 = or i1 undef, undef
  br i1 %cond, label %if.then44, label %if.end51

if.then44:                                        ; preds = %for.cond
  tail call void @bar() #0
  br label %if.end51

if.end51:                                         ; preds = %if.then44, %for.cond
  %.b433 = load i1, ptr @debug, align 4
  %or.cond290 = and i1 %or.cond288, %.b433
  br i1 %or.cond290, label %if.then55, label %if.end63

if.then55:                                        ; preds = %if.end51
  unreachable

if.end63:                                         ; preds = %if.end51
  unreachable
}

declare void @bar() local_unnamed_addr #0

attributes #0 = { nounwind }
