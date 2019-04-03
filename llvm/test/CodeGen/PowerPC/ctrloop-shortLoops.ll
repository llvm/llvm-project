; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -verify-machineinstrs -mcpu=pwr8 | FileCheck %s --check-prefixes=CHECK,CHECK-PWR8
; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -verify-machineinstrs -mcpu=a2q | FileCheck %s --check-prefixes=CHECK,CHECK-A2Q

; Verify that we do NOT generate the mtctr instruction for loop trip counts < 4
; The latency of the mtctr is only justified if there are more than 4 comparisons that are removed as a result.

@a = common local_unnamed_addr global i32 0, align 4
@b = common local_unnamed_addr global i32 0, align 4
@c = common local_unnamed_addr global i32 0, align 4
@d = common local_unnamed_addr global i32 0, align 4
@e = common local_unnamed_addr global i32 0, align 4
@f = common local_unnamed_addr global i32 0, align 4
@arr = common local_unnamed_addr global [5 x i32] zeroinitializer, align 4

; Function Attrs: norecurse nounwind readonly
define signext i32 @testTripCount2(i32 signext %a) {

; CHECK-LABEL: testTripCount2:
; CHECK-NOT: mtctr
; CHECK: blr

entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i32 %add

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 1, %entry ], [ %indvars.iv.next, %for.body ]
  %Sum.05 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds [5 x i32], [5 x i32]* @arr, i64 0, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, %Sum.05
  %indvars.iv.next = add nsw i64 %indvars.iv, -1
  %tobool = icmp eq i64 %indvars.iv, 0
  br i1 %tobool, label %for.cond.cleanup, label %for.body
}

; Function Attrs: norecurse nounwind readonly
define signext i32 @testTripCount3(i32 signext %a) {

; CHECK-LABEL: testTripCount3:
; CHECK-NOT: mtctr
; CHECK: blr

entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i32 %add

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 2, %entry ], [ %indvars.iv.next, %for.body ]
  %Sum.05 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds [5 x i32], [5 x i32]* @arr, i64 0, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, %Sum.05
  %indvars.iv.next = add nsw i64 %indvars.iv, -1
  %tobool = icmp eq i64 %indvars.iv, 0
  br i1 %tobool, label %for.cond.cleanup, label %for.body
}

; Function Attrs: norecurse nounwind readonly

define signext i32 @testTripCount4(i32 signext %a) {

; CHECK-LABEL: testTripCount4:
; CHECK: mtctr
; CHECK: bdnz

entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i32 %add

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 3, %entry ], [ %indvars.iv.next, %for.body ]
  %Sum.05 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds [5 x i32], [5 x i32]* @arr, i64 0, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, %Sum.05
  %indvars.iv.next = add nsw i64 %indvars.iv, -1
  %tobool = icmp eq i64 %indvars.iv, 0
  br i1 %tobool, label %for.cond.cleanup, label %for.body
}

; Function Attrs: norecurse nounwind
define signext i32 @testTripCount2NonSmallLoop() {
; CHECK-LABEL: testTripCount2NonSmallLoop:
; CHECK: bge
; CHECK: blr

entry:
  %.pre = load i32, i32* @a, align 4
  br label %for.body

for.body:                                         ; preds = %entry, %if.end
  %0 = phi i32 [ %.pre, %entry ], [ %1, %if.end ]
  %dec4 = phi i32 [ 1, %entry ], [ %dec, %if.end ]
  %b.03 = phi i8 [ 0, %entry ], [ %b.1, %if.end ]
  %tobool1 = icmp eq i32 %0, 0
  br i1 %tobool1, label %if.end, label %if.then

if.then:                                          ; preds = %for.body
  store i32 2, i32* @a, align 4
  br label %if.end

if.end:                                           ; preds = %for.body, %if.then
  %1 = phi i32 [ 2, %if.then ], [ 0, %for.body ]
  %b.1 = phi i8 [ 2, %if.then ], [ %b.03, %for.body ]
  %dec = add nsw i32 %dec4, -1
  %tobool = icmp eq i32 %dec4, 0
  br i1 %tobool, label %for.end, label %for.body

for.end:                                          ; preds = %if.end
  %conv = zext i8 %b.1 to i32
  ret i32 %conv
}

; On core a2q, IssueWidth is 1. On core pwr8, IssueWidth is 8.
; a2q should use mtctr, but pwr8 should not use mtctr.
define signext i32 @testTripCount5() {
; CHECK-LABEL: testTripCount5:
; CHECK-PWR8: mtctr
; CHECK-A2Q: mtctr
 
entry:
  %.prea = load i32, i32* @a, align 4
  %.preb = load i32, i32* @b, align 4
  %.prec = load i32, i32* @c, align 4
  %.pred = load i32, i32* @d, align 4
  %.pree = load i32, i32* @e, align 4
  %.pref = load i32, i32* @f, align 4
  br label %for.body

for.body:                                 ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 2, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = phi i32 [ %.prea, %entry ], [ %6, %for.body ]
  %1 = phi i32 [ %.preb, %entry ], [ %7, %for.body ]
  %2 = phi i32 [ %.prec, %entry ], [ %8, %for.body ]
  %3 = phi i32 [ %.pred, %entry ], [ %9, %for.body ]
  %4 = phi i32 [ %.pree, %entry ], [ %10, %for.body ]
  %5 = phi i32 [ %.pref, %entry ], [ %11, %for.body ]
  %6 = add i32 %0, 1
  %7 = add i32 %1, 1
  %8 = add i32 %2, 1
  %9 = add i32 %3, 1
  %10 = add i32 %4, 1
  %11 = add i32 %5, 1
  %indvars.iv.next = add nsw i64 %indvars.iv, -1
  %tobool = icmp eq i64 %indvars.iv, 0
  br i1 %tobool, label %for.end, label %for.body

for.end:                                 ; preds = %for.body
  store i32 %6, i32* @a, align 4 
  store i32 %7, i32* @b, align 4 
  store i32 %8, i32* @c, align 4 
  store i32 %9, i32* @d, align 4 
  store i32 %10, i32* @e, align 4 
  store i32 %11, i32* @f, align 4 
  ret i32 0
}

