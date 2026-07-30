; RUN: llc -mtriple=arm64-apple-ios -mcpu=cyclone < %s | FileCheck %s
; Do not generate redundant select in early if-converstion pass. 

define i32 @foo(i32 %a, i32 %b)  {
entry:
;CHECK-LABEL: foo:
;CHECK: csinc
;CHECK-NOT: csel
  %sub = sub nsw i32 %b, %a
  %cmp10 = icmp sgt i32 %a, 0
  br i1 %cmp10, label %while.body.lr.ph, label %while.end

while.body.lr.ph:
  br label %while.body

while.body:                                  
  %j.012 = phi i32 [ %sub, %while.body.lr.ph ], [ %inc, %if.then ], [ %inc, %if.else ]
  %i.011 = phi i32 [ %a, %while.body.lr.ph ], [ %inc2, %if.then ], [ %dec, %if.else ]
  %cmp1 = icmp slt i32 %i.011, %j.012
  br i1 %cmp1, label %while.end, label %while.cond

while.cond:
  %inc = add nsw i32 %j.012, 5
  %cmp2 = icmp slt i32 %inc, %b
  br i1 %cmp2, label %if.then, label %if.else

if.then:
  %inc2 = add nsw i32 %i.011, 1
  br label %while.body

if.else:
  %dec = add nsw i32 %i.011, -1
  br label %while.body

while.end:
  %j.0.lcssa = phi i32 [ %j.012, %while.body ], [ %sub, %entry ]
  %i.0.lcssa = phi i32 [ %i.011, %while.body ], [ %a, %entry ]
  %add = add nsw i32 %j.0.lcssa, %i.0.lcssa
  ret i32 %add
}

