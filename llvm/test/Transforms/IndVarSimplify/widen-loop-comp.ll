; RUN: opt < %s -indvars -S | FileCheck %s
target triple = "aarch64--linux-gnu"

; Provide legal integer types.
target datalayout = "n8:16:32:64"


; Check the loop exit i32 compare instruction and operand are widened to i64
; instead of truncating IV before its use in the i32 compare instruction.

@idx = common global i32 0, align 4
@e = common global i32 0, align 4
@ptr = common global i32* null, align 8

; CHECK-LABEL: @test1
; CHECK: for.body.lr.ph:
; CHECK: sext i32
; CHECK: for.cond:
; CHECK: icmp slt i64
; CHECK: for.body:
; CHECK: phi i64

define i32 @test1() {
entry:
  store i32 -1, i32* @idx, align 4
  %0 = load i32, i32* @e, align 4
  %cmp4 = icmp slt i32 %0, 0
  br i1 %cmp4, label %for.end.loopexit, label %for.body.lr.ph

for.body.lr.ph:
  %1 = load i32*, i32** @ptr, align 8
  %2 = load i32, i32* @e, align 4
  br label %for.body

for.cond:
  %inc = add nsw i32 %i.05, 1
  %cmp = icmp slt i32 %i.05, %2
  br i1 %cmp, label %for.body, label %for.cond.for.end.loopexit_crit_edge

for.body:
  %i.05 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.cond ]
  %idxprom = sext i32 %i.05 to i64
  %arrayidx = getelementptr inbounds i32, i32* %1, i64 %idxprom
  %3 = load i32, i32* %arrayidx, align 4
  %tobool = icmp eq i32 %3, 0
  br i1 %tobool, label %if.then, label %for.cond

if.then:
  %i.05.lcssa = phi i32 [ %i.05, %for.body ]
  store i32 %i.05.lcssa, i32* @idx, align 4
  br label %for.end

for.cond.for.end.loopexit_crit_edge:
  br label %for.end.loopexit

for.end.loopexit:
  br label %for.end

for.end:
  %4 = load i32, i32* @idx, align 4
  ret i32 %4
}

; CHECK-LABEL: @test2
; CHECK: for.body4.us
; CHECK: %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK: %cmp2.us = icmp ult i64
; CHECK-NOT: %2 = trunc i64 %indvars.iv.next to i32
; CHECK-NOT: %cmp2.us = icmp slt i32

define void @test2([8 x i8]* %a, i8* %b, i8 %limit) {
entry:
  %conv = zext i8 %limit to i32
  br i1 undef, label %for.cond1.preheader, label %for.cond1.preheader.us

for.cond1.preheader.us:
  %storemerge5.us = phi i32 [ 0, %entry ], [ %inc14.us, %for.inc13.us ]
  br i1 true, label %for.body4.lr.ph.us, label %for.inc13.us

for.inc13.us:
  %inc14.us = add nsw i32 %storemerge5.us, 1
  %cmp.us = icmp slt i32 %inc14.us, 4
  br i1 %cmp.us, label %for.cond1.preheader.us, label %for.end

for.body4.us:
  %storemerge14.us = phi i32 [ 0, %for.body4.lr.ph.us ], [ %inc.us, %for.body4.us ]
  %idxprom.us = sext i32 %storemerge14.us to i64
  %arrayidx6.us = getelementptr inbounds [8 x i8], [8 x i8]* %a, i64 %idxprom5.us, i64 %idxprom.us
  %0 = load i8, i8* %arrayidx6.us, align 1
  %idxprom7.us = zext i8 %0 to i64
  %arrayidx8.us = getelementptr inbounds i8, i8* %b, i64 %idxprom7.us
  %1 = load i8, i8* %arrayidx8.us, align 1
  store i8 %1, i8* %arrayidx6.us, align 1
  %inc.us = add nsw i32 %storemerge14.us, 1
  %cmp2.us = icmp slt i32 %inc.us, %conv
  br i1 %cmp2.us, label %for.body4.us, label %for.inc13.us

for.body4.lr.ph.us:
  %idxprom5.us = sext i32 %storemerge5.us to i64
  br label %for.body4.us

for.cond1.preheader:
  %storemerge5 = phi i32 [ 0, %entry ], [ %inc14, %for.inc13 ]
  br i1 false, label %for.inc13, label %for.inc13

for.inc13:
  %inc14 = add nsw i32 %storemerge5, 1
  %cmp = icmp slt i32 %inc14, 4
  br i1 %cmp, label %for.cond1.preheader, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: @test3
; CHECK: sext i32 %b
; CHECK: for.cond:
; CHECK: phi i64
; CHECK: icmp slt i64

define i32 @test3(i32* %a, i32 %b) {
entry:
  br label %for.cond

for.cond:
  %sum.0 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp slt i32 %i.0, %b
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %idxprom = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %sum.0, %0
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret i32 %sum.0
}

declare i32 @fn1(i8 signext)

; PR21030
; CHECK-LABEL: @test4
; CHECK: for.body:
; CHECK: phi i32
; CHECK: icmp sgt i8

define i32 @test4(i32 %a) {
entry:
  br label %for.body

for.body:
  %c.07 = phi i8 [ -3, %entry ], [ %dec, %for.body ]
  %conv6 = zext i8 %c.07 to i32
  %or = or i32 %a, %conv6
  %conv3 = trunc i32 %or to i8
  %call = call i32 @fn1(i8 signext %conv3)
  %dec = add i8 %c.07, -1
  %cmp = icmp sgt i8 %dec, -14
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret i32 0
}

; CHECK-LABEL: @test5
; CHECK: zext i32 %b
; CHECK: for.cond:
; CHECK: phi i64
; CHECK: icmp ule i64

define i32 @test5(i32* %a, i32 %b) {
entry:
  br label %for.cond

for.cond:
  %sum.0 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp ule i32 %i.0, %b
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %idxprom = zext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %sum.0, %0
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret i32 %sum.0
}

define i32 @test6(i32* %a, i32 %b) {
; CHECK-LABEL: @test6(
; CHECK: [[B_SEXT:%[a-z0-9]+]] = sext i32 %b to i64
; CHECK: for.cond:
; CHECK: icmp sle i64 %indvars.iv, [[B_SEXT]]

entry:
  br label %for.cond

for.cond:
  %sum.0 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp sle i32 %i.0, %b
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %idxprom = zext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %sum.0, %0
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret i32 %sum.0
}

define i32 @test7(i32* %a, i32 %b) {
; CHECK-LABEL: @test7(
; CHECK: [[B_ZEXT:%[a-z0-9]+]] = zext i32 %b to i64
; CHECK: [[B_SEXT:%[a-z0-9]+]] = sext i32 %b to i64
; CHECK: for.cond:
; CHECK: icmp ule i64 %indvars.iv, [[B_ZEXT]]
; CHECK: for.body:
; CHECK: icmp sle i64 %indvars.iv, [[B_SEXT]]

entry:
  br label %for.cond

for.cond:
  %sum.0 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp ule i32 %i.0, %b
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %idxprom = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %sum.0, %0
  %inc = add nsw i32 %i.0, 1
  %cmp2 = icmp sle i32 %i.0, %b
  br i1 %cmp2, label %for.cond, label %for.end

for.end:
  ret i32 %sum.0
}

define i32 @test8(i32* %a, i32 %b, i32 %init) {
; CHECK-LABEL: @test8(
; CHECK: [[INIT_SEXT:%[a-z0-9]+]] = sext i32 %init to i64
; CHECK: [[B_ZEXT:%[a-z0-9]+]] = zext i32 %b to i64
; CHECK: for.cond:
;     Note: %indvars.iv is the sign extension of %i.0
; CHECK: %indvars.iv = phi i64 [ [[INIT_SEXT]], %for.cond.preheader ], [ %indvars.iv.next, %for.body ]
; CHECK: icmp ule i64 %indvars.iv, [[B_ZEXT]]

entry:
  %e = icmp sgt i32 %init, 0
  br i1 %e, label %for.cond, label %leave

for.cond:
  %sum.0 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %i.0 = phi i32 [ %init, %entry ], [ %inc, %for.body ]
  %cmp = icmp ule i32 %i.0, %b
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %idxprom = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %sum.0, %0
  %inc = add nsw i32 %i.0, 1
  %cmp2 = icmp slt i32 0, %inc
  br i1 %cmp2, label %for.cond, label %for.end

for.end:
  ret i32 %sum.0

leave:
  ret i32 0
}

define i32 @test9(i32* %a, i32 %b, i32 %init) {
; CHECK-LABEL: @test9(
; CHECK: [[INIT_ZEXT:%[a-z0-9]+]] = zext i32 %init to i64
; CHECK: [[B_SEXT:%[a-z0-9]+]] = sext i32 %b to i64
; CHECK: for.cond:
;     Note: %indvars.iv is the zero extension of %i.0
; CHECK: %indvars.iv = phi i64 [ [[INIT_ZEXT]], %for.cond.preheader ], [ %indvars.iv.next, %for.body ]
; CHECK: icmp slt i64 %indvars.iv, [[B_SEXT]]

entry:
  %e = icmp sgt i32 %init, 0
  br i1 %e, label %for.cond, label %leave

for.cond:
  %sum.0 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %i.0 = phi i32 [ %init, %entry ], [ %inc, %for.body ]
  %cmp = icmp slt i32 %i.0, %b
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %idxprom = zext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %sum.0, %0
  %inc = add nsw i32 %i.0, 1
  %cmp2 = icmp slt i32 0, %inc
  br i1 %cmp2, label %for.cond, label %for.end

for.end:
  ret i32 %sum.0

leave:
  ret i32 0
}

declare void @consume.i64(i64)
declare void @consume.i1(i1)

define i32 @test10(i32 %v) {
; CHECK-LABEL: @test10(
 entry:
; CHECK-NOT: zext
  br label %loop

 loop:
; CHECK: [[WIDE_V:%[a-z0-9]+]] = sext i32 %v to i64
; CHECK: loop:
; CHECK: %indvars.iv = phi i64 [ %indvars.iv.next, %loop ], [ 0, %entry ]
; CHECK: %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK: [[MUL:%[a-z0-9]+]] = mul nsw i64 %indvars.iv, -1
; CHECK: [[CMP:%[a-z0-9]+]] = icmp eq i64 [[MUL]], [[WIDE_V]]
; CHECK: call void @consume.i1(i1 [[CMP]])

  %i = phi i32 [ 0, %entry ], [ %i.inc, %loop ]
  %i.inc = add i32 %i, 1
  %iv = mul i32 %i, -1
  %cmp = icmp eq i32 %iv, %v
  call void @consume.i1(i1 %cmp)
  %be.cond = icmp slt i32 %i.inc, 11
  %ext = sext i32 %iv to i64
  call void @consume.i64(i64 %ext)
  br i1 %be.cond, label %loop, label %leave

 leave:
  ret i32 22
}
