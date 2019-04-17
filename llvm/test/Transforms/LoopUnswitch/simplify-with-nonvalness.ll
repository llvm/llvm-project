; RUN: opt < %s -loop-unswitch -verify-loop-info -S < %s 2>&1 | FileCheck %s
; RUN: opt < %s -loop-unswitch -verify-loop-info -enable-mssa-loop-dependency=true -verify-memoryssa -S < %s 2>&1 | FileCheck %s

; There are 1 case and 1 default case in the switch. after we unswitch, we know the
; %a is definitely not 0 in one of the unswitched loop, make sure we take advantage
; of that and simplify the branches in the loop.
;
; CHECK: define void @simplify_with_nonvalness(

; This is the loop in which we know %a is definitely 0.
; CHECK: sw.bb.us:
; CHECK: br i1 true, label %if.then.us, label %if.end.us

; This is the loop in which we do not know what %a is but we know %a is definitely NOT 0.
; Make sure we use that information to simplify.
; The icmp eq i32 %a, 0 in one of the unswitched loop is simplified to false.
; CHECK: sw.bb.split:
; CHECK: br i1 false, label %if.then, label %if.end

define void @simplify_with_nonvalness(i32 %a) #0 {
entry:
  br label %for.cond

for.cond:
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, 1024
  br i1 %cmp, label %for.body, label %for.end

for.body:
  switch i32 %a, label %sw.default [
    i32 0, label %sw.bb
  ]

sw.bb:
  %cmp1 = icmp eq i32 %a, 0
  br i1 %cmp1, label %if.then, label %if.end

if.then:
  call void (...) @bar()
  br label %if.end

if.end:
  br label %sw.epilog

sw.default:
  br label %sw.epilog

sw.epilog:
  br label %for.inc

for.inc:
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret void
}

declare void @bar(...) 
