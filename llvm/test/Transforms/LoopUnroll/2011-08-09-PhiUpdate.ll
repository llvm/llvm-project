; RUN: opt -S < %s -instcombine -inline -jump-threading -loop-unroll -unroll-count=4 | FileCheck %s
;
; This is a test case that required a number of setup passes because
; it depends on block order.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.6.8"

declare i1 @check() nounwind
declare i32 @getval() nounwind

; Check that the loop exit merges values from all the iterations. This
; could be a tad fragile, but it's a good test.
;
; CHECK-LABEL: @foo(
; CHECK: return:
; CHECK: %retval.0 = phi i32 [ %tmp7.i, %land.lhs.true ], [ 0, %do.cond ], [ %tmp7.i.1, %land.lhs.true.1 ], [ 0, %do.cond.1 ], [ %tmp7.i.2, %land.lhs.true.2 ], [ 0, %do.cond.2 ], [ %tmp7.i.3, %land.lhs.true.3 ], [ 0, %do.cond.3 ]
; CHECK-NOT: @bar(
; CHECK: bar.exit.3
define i32 @foo() uwtable ssp align 2 {
entry:
  br i1 undef, label %return, label %if.end

if.end:                                           ; preds = %entry
  %call2 = call i32 @getval()
  br label %do.body

do.body:                                          ; preds = %do.cond, %if.end
  %call6 = call i32 @bar()
  %cmp = icmp ne i32 %call6, 0
  br i1 %cmp, label %land.lhs.true, label %do.cond

land.lhs.true:                                    ; preds = %do.body
  %call10 = call i32 @getval()
  %cmp11 = icmp eq i32 0, %call10
  br i1 %cmp11, label %return, label %do.cond

do.cond:                                          ; preds = %land.lhs.true, %do.body
  %cmp18 = icmp sle i32 0, %call2
  br i1 %cmp18, label %do.body, label %return

return:                                           ; preds = %do.cond, %land.lhs.true, %entry
  %retval.0 = phi i32 [ 0, %entry ], [ %call6, %land.lhs.true ], [ 0, %do.cond ]
  ret i32 %retval.0
}

define linkonce_odr i32 @bar() nounwind uwtable ssp align 2 {
entry:
  br i1 undef, label %land.lhs.true, label %cond.end

land.lhs.true:                                    ; preds = %entry
  %cmp4 = call zeroext i1 @check()
  br i1 %cmp4, label %cond.true, label %cond.end

cond.true:                                        ; preds = %land.lhs.true
  %tmp7 = call i32 @getval()
  br label %cond.end

cond.end:                                         ; preds = %cond.true, %land.lhs.true, %entry
  %cond = phi i32 [ %tmp7, %cond.true ], [ 0, %land.lhs.true ], [ 0, %entry ]
  ret i32 %cond
}
