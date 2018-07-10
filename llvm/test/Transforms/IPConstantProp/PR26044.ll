; RUN: opt < %s -S -ipsccp | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @fn2() {
entry:
  br label %if.end

for.cond1:                                        ; preds = %if.end, %for.end
  br i1 undef, label %if.end, label %if.end

if.end:                                           ; preds = %lbl, %for.cond1
  %e.2 = phi i32* [ undef, %entry ], [ null, %for.cond1 ], [ null, %for.cond1 ]
  %0 = load i32, i32* %e.2, align 4
  %call = call i32 @fn1(i32 %0)
  br label %for.cond1
}

define internal i32 @fn1(i32 %p1) {
entry:
  %tobool = icmp ne i32 %p1, 0
  %cond = select i1 %tobool, i32 %p1, i32 %p1
  ret i32 %cond
}

define void @fn_no_null_opt() #0 {
entry:
  br label %if.end

for.cond1:                                        ; preds = %if.end, %for.end
  br i1 undef, label %if.end, label %if.end

if.end:                                           ; preds = %lbl, %for.cond1
  %e.2 = phi i32* [ undef, %entry ], [ null, %for.cond1 ], [ null, %for.cond1 ]
  %0 = load i32, i32* %e.2, align 4
  %call = call i32 @fn0(i32 %0)
  br label %for.cond1
}

define internal i32 @fn0(i32 %p1) {
entry:
  %tobool = icmp ne i32 %p1, 0
  %cond = select i1 %tobool, i32 %p1, i32 %p1
  ret i32 %cond
}

attributes #0 = { "null-pointer-is-valid"="true" }

; CHECK-LABEL: define void @fn2(
; CHECK: call i32 @fn1(i32 undef)

; CHECK-LABEL: define internal i32 @fn1(
; CHECK:%[[COND:.*]] = select i1 undef, i32 undef, i32 undef
; CHECK: ret i32 %[[COND]]

; CHECK-LABEL: define void @fn_no_null_opt(
; CHECK: call i32 @fn0(i32 %0)

; CHECK-LABEL: define internal i32 @fn0(
; CHECK:%[[TOBOOL:.*]] = icmp ne i32 %p1, 0
; CHECK:%[[COND:.*]] = select i1 %[[TOBOOL]], i32 %p1, i32 %p1
; CHECK: ret i32 %[[COND]]
