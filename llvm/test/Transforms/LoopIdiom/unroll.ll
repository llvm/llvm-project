; RUN: opt -basicaa -loop-idiom < %s -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

; CHECK: @.memset_pattern = private unnamed_addr constant [4 x i32] [i32 2, i32 2, i32 2, i32 2], align 16

target triple = "x86_64-apple-darwin10.0.0"

;void test(int *f, unsigned n) {
;  for (unsigned i = 0; i < 2 * n; i += 2) {
;    f[i] = 0;
;    f[i+1] = 0;
;  }
;}
define void @test(i32* %f, i32 %n) nounwind ssp {
entry:
  %mul = shl i32 %n, 1
  %cmp1 = icmp eq i32 %mul, 0
  br i1 %cmp1, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %0 = zext i32 %mul to i64
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %f, i64 %indvars.iv
  store i32 0, i32* %arrayidx, align 4
  %1 = or i64 %indvars.iv, 1
  %arrayidx2 = getelementptr inbounds i32, i32* %f, i64 %1
  store i32 0, i32* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 2
  %cmp = icmp ult i64 %indvars.iv.next, %0
  br i1 %cmp, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
; CHECK-LABEL: @test(
; CHECK: call void @llvm.memset
; CHECK-NOT: store
}

;void test_pattern(int *f, unsigned n) {
;  for (unsigned i = 0; i < 2 * n; i += 2) {
;    f[i] = 2;
;    f[i+1] = 2;
;  }
;}
define void @test_pattern(i32* %f, i32 %n) nounwind ssp {
entry:
  %mul = shl i32 %n, 1
  %cmp1 = icmp eq i32 %mul, 0
  br i1 %cmp1, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %0 = zext i32 %mul to i64
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %f, i64 %indvars.iv
  store i32 2, i32* %arrayidx, align 4
  %1 = or i64 %indvars.iv, 1
  %arrayidx2 = getelementptr inbounds i32, i32* %f, i64 %1
  store i32 2, i32* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 2
  %cmp = icmp ult i64 %indvars.iv.next, %0
  br i1 %cmp, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
; CHECK-LABEL: @test_pattern(
; CHECK: call void @memset_pattern16
; CHECK-NOT: store
}
