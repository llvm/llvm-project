; RUN: opt < %s -bounds-checking -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK-LABEL: @sumSize
define dso_local i32 @sumSize(i32 %n) {
entry:
  %foo = alloca [1000 x i32], align 16
  %0 = bitcast [1000 x i32]* %foo to i8*
  call void @llvm.lifetime.start.p0i8(i64 4000, i8* nonnull %0)
  %arraydecay = getelementptr inbounds [1000 x i32], [1000 x i32]* %foo, i64 0, i64 0
  call void @fill(i32* nonnull %arraydecay, i32 %n)
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %entry
  %indvars.iv.i = phi i64 [ 0, %entry ], [ %indvars.iv.next.i, %for.body.i ]
  %sum.07.i = phi i32 [ 0, %entry ], [ %add.i, %for.body.i ]
  %arrayidx.i = getelementptr inbounds [1000 x i32], [1000 x i32]* %foo, i64 0, i64 %indvars.iv.i
; CHECK-NOT: trap
  %1 = load i32, i32* %arrayidx.i, align 4
  %add.i = add nsw i32 %1, %sum.07.i
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %exitcond.i = icmp eq i64 %indvars.iv.next.i, 1000
  br i1 %exitcond.i, label %accumulate.exit, label %for.body.i

accumulate.exit:                                  ; preds = %for.body.i
  call void @llvm.lifetime.end.p0i8(i64 4000, i8* nonnull %0)
  ret i32 %add.i
}

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)

declare dso_local void @fill(i32*, i32)

declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)

; CHECK-LABEL: @sumSizePlusOne
define dso_local i32 @sumSizePlusOne(i32 %n) {
entry:
  %foo = alloca [1000 x i32], align 16
  %0 = bitcast [1000 x i32]* %foo to i8*
  call void @llvm.lifetime.start.p0i8(i64 4000, i8* nonnull %0)
  %arraydecay = getelementptr inbounds [1000 x i32], [1000 x i32]* %foo, i64 0, i64 0
  call void @fill(i32* nonnull %arraydecay, i32 %n)
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %entry
  %indvars.iv.i = phi i64 [ 0, %entry ], [ %indvars.iv.next.i, %for.body.i ]
  %sum.01.i = phi i32 [ 0, %entry ], [ %add.i, %for.body.i ]
  %arrayidx.i = getelementptr inbounds [1000 x i32], [1000 x i32]* %foo, i64 0, i64 %indvars.iv.i
; CHECK: mul i64 {{.*}}, 4
; CHECK: sub i64 4000, %
; CHECK-NEXT: icmp ult i64 {{.*}}, 4
; CHECK-NEXT: or i1
; CHECK: trap
  %1 = load i32, i32* %arrayidx.i, align 4
  %add.i = add nsw i32 %1, %sum.01.i
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %exitcond.i = icmp eq i64 %indvars.iv.next.i, 1001
  br i1 %exitcond.i, label %accumulate.exit, label %for.body.i

accumulate.exit:                                  ; preds = %for.body.i
  call void @llvm.lifetime.end.p0i8(i64 4000, i8* nonnull %0)
  ret i32 %add.i
}

; CHECK-LABEL: @sumLarger
define dso_local i32 @sumLarger(i32 %n) {
entry:
  %foo = alloca [1000 x i32], align 16
  %0 = bitcast [1000 x i32]* %foo to i8*
  call void @llvm.lifetime.start.p0i8(i64 4000, i8* nonnull %0)
  %arraydecay = getelementptr inbounds [1000 x i32], [1000 x i32]* %foo, i64 0, i64 0
  call void @fill(i32* nonnull %arraydecay, i32 %n)
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %entry
  %indvars.iv.i = phi i64 [ 0, %entry ], [ %indvars.iv.next.i, %for.body.i ]
  %sum.07.i = phi i32 [ 0, %entry ], [ %add.i, %for.body.i ]
  %arrayidx.i = getelementptr inbounds [1000 x i32], [1000 x i32]* %foo, i64 0, i64 %indvars.iv.i
; CHECK: mul i64 {{.*}}, 4
; CHECK: sub i64 4000, %
; CHECK-NEXT: icmp ult i64 4000, %
; CHECK-NEXT: icmp ult i64 {{.*}}, 4
; CHECK-NEXT: or i1
; CHECK: trap
  %1 = load i32, i32* %arrayidx.i, align 4
  %add.i = add nsw i32 %1, %sum.07.i
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %exitcond.i = icmp eq i64 %indvars.iv.next.i, 2000
  br i1 %exitcond.i, label %accumulate.exit, label %for.body.i

accumulate.exit:                                  ; preds = %for.body.i
  call void @llvm.lifetime.end.p0i8(i64 4000, i8* nonnull %0)
  ret i32 %add.i
}

; CHECK-LABEL: @sumUnknown
define dso_local i32 @sumUnknown(i32 %n) {
entry:
  %foo = alloca [1000 x i32], align 16
  %0 = bitcast [1000 x i32]* %foo to i8*
  call void @llvm.lifetime.start.p0i8(i64 4000, i8* nonnull %0)
  %arraydecay = getelementptr inbounds [1000 x i32], [1000 x i32]* %foo, i64 0, i64 0
  call void @fill(i32* nonnull %arraydecay, i32 %n)
  %cmp6.i = icmp eq i32 %n, 0
  br i1 %cmp6.i, label %accumulate.exit, label %for.body.preheader.i

for.body.preheader.i:                             ; preds = %entry
  %wide.trip.count.i = zext i32 %n to i64
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %for.body.preheader.i
  %indvars.iv.i = phi i64 [ 0, %for.body.preheader.i ], [ %indvars.iv.next.i, %for.body.i ]
  %sum.07.i = phi i32 [ 0, %for.body.preheader.i ], [ %add.i, %for.body.i ]
  %arrayidx.i = getelementptr inbounds [1000 x i32], [1000 x i32]* %foo, i64 0, i64 %indvars.iv.i
; CHECK: mul i64 {{.*}}, 4
; CHECK: sub i64 4000, %
; CHECK-NEXT: icmp ult i64 4000, %
; CHECK-NEXT: icmp ult i64 {{.*}}, 4
; CHECK-NEXT: or i1
; CHECK: trap
  %1 = load i32, i32* %arrayidx.i, align 4
  %add.i = add nsw i32 %1, %sum.07.i
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %exitcond.i = icmp eq i64 %indvars.iv.next.i, %wide.trip.count.i
  br i1 %exitcond.i, label %accumulate.exit, label %for.body.i

accumulate.exit:                                  ; preds = %for.body.i, %entry
  %sum.0.lcssa.i = phi i32 [ 0, %entry ], [ %add.i, %for.body.i ]
  call void @llvm.lifetime.end.p0i8(i64 4000, i8* nonnull %0)
  ret i32 %sum.0.lcssa.i
}

; CHECK-LABEL: @twoDimSize
define dso_local i32 @twoDimSize(i32 %n) {
entry:
  %foo = alloca [2 x [2 x i32]], align 16
  %0 = bitcast [2 x [2 x i32]]* %foo to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %0)
  %arraydecay = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %foo, i64 0, i64 0, i64 0
  call void @fill(i32* nonnull %arraydecay, i32 %n)
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond.cleanup3, %entry
  %indvars.iv23 = phi i64 [ 0, %entry ], [ %indvars.iv.next24, %for.cond.cleanup3 ]
  %sum.021 = phi i32 [ 0, %entry ], [ %add, %for.cond.cleanup3 ]
  br label %for.body4

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %0)
  ret i32 %add

for.cond.cleanup3:                                ; preds = %for.body4
  %indvars.iv.next24 = add nuw nsw i64 %indvars.iv23, 1
  %exitcond25 = icmp eq i64 %indvars.iv.next24, 2
  br i1 %exitcond25, label %for.cond.cleanup, label %for.cond1.preheader

for.body4:                                        ; preds = %for.body4, %for.cond1.preheader
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.body4 ]
  %sum.119 = phi i32 [ %sum.021, %for.cond1.preheader ], [ %add, %for.body4 ]
  %arrayidx7 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %foo, i64 0, i64 %indvars.iv23, i64 %indvars.iv
; CHECK-NOT: trap
  %1 = load i32, i32* %arrayidx7, align 4
  %add = add nsw i32 %1, %sum.119
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 2
  br i1 %exitcond, label %for.cond.cleanup3, label %for.body4
}

; CHECK-LABEL: @twoDimLarger1
define dso_local i32 @twoDimLarger1(i32 %n) {
entry:
  %foo = alloca [2 x [2 x i32]], align 16
  %0 = bitcast [2 x [2 x i32]]* %foo to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %0)
  %arraydecay = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %foo, i64 0, i64 0, i64 0
  call void @fill(i32* nonnull %arraydecay, i32 %n)
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond.cleanup3, %entry
  %indvars.iv23 = phi i64 [ 0, %entry ], [ %indvars.iv.next24, %for.cond.cleanup3 ]
  %sum.021 = phi i32 [ 0, %entry ], [ %add, %for.cond.cleanup3 ]
  br label %for.body4

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %0)
  ret i32 %add

for.cond.cleanup3:                                ; preds = %for.body4
  %indvars.iv.next24 = add nuw nsw i64 %indvars.iv23, 1
  %exitcond25 = icmp eq i64 %indvars.iv.next24, 3
  br i1 %exitcond25, label %for.cond.cleanup, label %for.cond1.preheader

for.body4:                                        ; preds = %for.body4, %for.cond1.preheader
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.body4 ]
  %sum.119 = phi i32 [ %sum.021, %for.cond1.preheader ], [ %add, %for.body4 ]
  %arrayidx7 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %foo, i64 0, i64 %indvars.iv23, i64 %indvars.iv
; CHECK: mul i64 {{.*}}, 8
; CHECK: mul i64 {{.*}}, 4
; CHECK: add i64
; CHECK: sub i64 16, %
; CHECK-NEXT: icmp ult i64 16, %
; CHECK-NEXT: icmp ult i64 {{.*}}, 4
; CHECK-NEXT: or i1
; CHECK: trap
  %1 = load i32, i32* %arrayidx7, align 4
  %add = add nsw i32 %1, %sum.119
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 2
  br i1 %exitcond, label %for.cond.cleanup3, label %for.body4
}

; CHECK-LABEL: @twoDimLarger2
define dso_local i32 @twoDimLarger2(i32 %n) {
entry:
  %foo = alloca [2 x [2 x i32]], align 16
  %0 = bitcast [2 x [2 x i32]]* %foo to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %0)
  %arraydecay = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %foo, i64 0, i64 0, i64 0
  call void @fill(i32* nonnull %arraydecay, i32 %n)
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond.cleanup3, %entry
  %indvars.iv23 = phi i64 [ 0, %entry ], [ %indvars.iv.next24, %for.cond.cleanup3 ]
  %sum.021 = phi i32 [ 0, %entry ], [ %add, %for.cond.cleanup3 ]
  br label %for.body4

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %0)
  ret i32 %add

for.cond.cleanup3:                                ; preds = %for.body4
  %indvars.iv.next24 = add nuw nsw i64 %indvars.iv23, 1
  %exitcond25 = icmp eq i64 %indvars.iv.next24, 2
  br i1 %exitcond25, label %for.cond.cleanup, label %for.cond1.preheader

for.body4:                                        ; preds = %for.body4, %for.cond1.preheader
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.body4 ]
  %sum.119 = phi i32 [ %sum.021, %for.cond1.preheader ], [ %add, %for.body4 ]
  %arrayidx7 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %foo, i64 0, i64 %indvars.iv23, i64 %indvars.iv
; CHECK: mul i64 {{.*}}, 8
; CHECK: mul i64 {{.*}}, 4
; CHECK: add i64
; CHECK: sub i64 16, %
; CHECK-NEXT: icmp ult i64 {{.*}}, 4
; CHECK-NEXT: or i1
; CHECK: trap
  %1 = load i32, i32* %arrayidx7, align 4
  %add = add nsw i32 %1, %sum.119
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 3
  br i1 %exitcond, label %for.cond.cleanup3, label %for.body4
}

; CHECK-LABEL: @twoDimUnknown
define dso_local i32 @twoDimUnknown(i32 %n) {
entry:
  %foo = alloca [2 x [2 x i32]], align 16
  %0 = bitcast [2 x [2 x i32]]* %foo to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %0)
  %arraydecay = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %foo, i64 0, i64 0, i64 0
  call void @fill(i32* nonnull %arraydecay, i32 %n)
  %cmp24 = icmp eq i32 %n, 0
  br i1 %cmp24, label %for.cond.cleanup, label %for.cond1.preheader.lr.ph

for.cond1.preheader.lr.ph:                        ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  %wide.trip.count.le = zext i32 %n to i64
  br label %for.body4.lr.ph

for.body4.lr.ph:                                  ; preds = %for.cond1.preheader.lr.ph, %for.cond.cleanup3
  %indvars.iv28 = phi i64 [ 0, %for.cond1.preheader.lr.ph ], [ %indvars.iv.next29, %for.cond.cleanup3 ]
  %sum.025 = phi i32 [ 0, %for.cond1.preheader.lr.ph ], [ %add, %for.cond.cleanup3 ]
  br label %for.body4

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3, %entry
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.cond.cleanup3 ]
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %0)
  ret i32 %sum.0.lcssa

for.cond.cleanup3:                                ; preds = %for.body4
  %indvars.iv.next29 = add nuw nsw i64 %indvars.iv28, 1
  %exitcond31 = icmp eq i64 %indvars.iv.next29, %wide.trip.count.le
  br i1 %exitcond31, label %for.cond.cleanup, label %for.body4.lr.ph

for.body4:                                        ; preds = %for.body4, %for.body4.lr.ph
  %indvars.iv = phi i64 [ 0, %for.body4.lr.ph ], [ %indvars.iv.next, %for.body4 ]
  %sum.122 = phi i32 [ %sum.025, %for.body4.lr.ph ], [ %add, %for.body4 ]
  %arrayidx7 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %foo, i64 0, i64 %indvars.iv28, i64 %indvars.iv
; CHECK: mul i64 {{.*}}, 8
; CHECK: mul i64 {{.*}}, 4
; CHECK: add i64
; CHECK: sub i64 16, %
; CHECK-NEXT: icmp ult i64 16, %
; CHECK-NEXT: icmp ult i64 {{.*}}, 4
; CHECK-NEXT: or i1
; CHECK: trap
  %1 = load i32, i32* %arrayidx7, align 4
  %add = add nsw i32 %1, %sum.122
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.cond.cleanup3, label %for.body4
}

; CHECK-LABEL: @countDownGood
define dso_local i32 @countDownGood(i32 %n) {
entry:
  %foo = alloca [1000 x i32], align 16
  %0 = bitcast [1000 x i32]* %foo to i8*
  call void @llvm.lifetime.start.p0i8(i64 4000, i8* nonnull %0)
  %arraydecay = getelementptr inbounds [1000 x i32], [1000 x i32]* %foo, i64 0, i64 0
  call void @fill(i32* nonnull %arraydecay, i32 %n)
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  call void @llvm.lifetime.end.p0i8(i64 4000, i8* nonnull %0)
  ret i32 %add

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 999, %entry ], [ %indvars.iv.next, %for.body ]
  %sum.06 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds [1000 x i32], [1000 x i32]* %foo, i64 0, i64 %indvars.iv
; CHECK-NOT: trap
  %1 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %1, %sum.06
  %indvars.iv.next = add nsw i64 %indvars.iv, -1
  %cmp = icmp eq i64 %indvars.iv, 0
  br i1 %cmp, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: @countDownBad
define dso_local i32 @countDownBad(i32 %n) {
entry:
  %foo = alloca [1000 x i32], align 16
  %0 = bitcast [1000 x i32]* %foo to i8*
  call void @llvm.lifetime.start.p0i8(i64 4000, i8* nonnull %0)
  %arraydecay = getelementptr inbounds [1000 x i32], [1000 x i32]* %foo, i64 0, i64 0
  call void @fill(i32* nonnull %arraydecay, i32 %n)
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  call void @llvm.lifetime.end.p0i8(i64 4000, i8* nonnull %0)
  ret i32 %add

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 999, %entry ], [ %indvars.iv.next, %for.body ]
  %sum.06 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds [1000 x i32], [1000 x i32]* %foo, i64 0, i64 %indvars.iv
; CHECK: mul i64 {{.*}}, 4
; CHECK: sub i64 4000, %
; CHECK-NEXT: icmp ult i64 4000, %
; CHECK: trap
  %1 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %1, %sum.06
  %indvars.iv.next = add nsw i64 %indvars.iv, -1
  %cmp = icmp sgt i64 %indvars.iv, -1
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}
