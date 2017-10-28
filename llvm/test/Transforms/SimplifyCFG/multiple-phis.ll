; RUN: opt -simplifycfg -keep-loops=false -S < %s | FileCheck %s

; It's not worthwhile to if-convert one of the phi nodes and leave
; the other behind, because that still requires a branch. If
; SimplifyCFG if-converts one of the phis, it should do both.

; CHECK:      %div.high.addr.0 = select i1 %cmp1, i32 %div, i32 %high.addr.0
; CHECK-NEXT: %low.0.add2 = select i1 %cmp1, i32 %low.0, i32 %add2
; CHECK-NEXT: br label %while.cond

define i32 @upper_bound(i32* %r, i32 %high, i32 %k) nounwind {
entry:
  br label %while.cond

while.cond:                                       ; preds = %if.then, %if.else, %entry
  %high.addr.0 = phi i32 [ %high, %entry ], [ %div, %if.then ], [ %high.addr.0, %if.else ]
  %low.0 = phi i32 [ 0, %entry ], [ %low.0, %if.then ], [ %add2, %if.else ]
  %cmp = icmp ult i32 %low.0, %high.addr.0
  br i1 %cmp, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  %add = add i32 %low.0, %high.addr.0
  %div = udiv i32 %add, 2
  %idxprom = zext i32 %div to i64
  %arrayidx = getelementptr inbounds i32, i32* %r, i64 %idxprom
  %0 = load i32, i32* %arrayidx
  %cmp1 = icmp ult i32 %k, %0
  br i1 %cmp1, label %if.then, label %if.else

if.then:                                          ; preds = %while.body
  br label %while.cond

if.else:                                          ; preds = %while.body
  %add2 = add i32 %div, 1
  br label %while.cond

while.end:                                        ; preds = %while.cond
  ret i32 %low.0
}
