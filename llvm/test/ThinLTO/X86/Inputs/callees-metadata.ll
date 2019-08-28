target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@fptr = internal unnamed_addr global i32 (i32)* @f2, align 8

define dso_local i32 @foo(i32 %x) local_unnamed_addr {
entry:
  %0 = load i32 (i32)*, i32 (i32)** @fptr, align 8
  %call = tail call i32 %0(i32 %x), !callees !0
  ret i32 %call
}

define internal i32 @f2(i32 %x) {
entry:
  %tobool = icmp eq i32 %x, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  store i32 (i32)* @f1, i32 (i32)** @fptr, align 8
  %sub.i = add nsw i32 %x, -1
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %phi.call = phi i32 [ %sub.i, %if.then ], [ -1, %entry ]
  ret i32 %phi.call
}

define internal i32 @f1(i32 %x) {
entry:
  %sub = add nsw i32 %x, -1
  ret i32 %sub
}

!0 = !{i32 (i32)* @f1, i32 (i32)* @f2}
