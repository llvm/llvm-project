; REQUIRES: asserts
; RUN: opt -mtriple=s390x-unknown-linux -mcpu=z16 -passes=loop-vectorize \
; RUN:   -debug-only=loop-vectorize -force-vector-width=4 \
; RUN:   -disable-output < %s 2>&1 | FileCheck %s
;
; Check cost function for <8 x i128> store interleave group.

; CHECK: LV: Checking a loop in 'fun'
; CHECK: LV: Found an estimated cost of 4 for VF 4 For instruction:   store i128 8721036757475490113
; CHECK: LV: Found an estimated cost of 4 for VF 4 For instruction:   store i128 8721036757475490113

define noundef i32 @fun(i32 %argc, ptr nocapture readnone %argv) {
entry:
  %l_4774.i = alloca [4 x [2 x i128]], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %l_4774.i)
  br label %for.cond4.preheader.i

for.cond4.preheader.i:                            ; preds = %for.cond4.preheader.i, %entry
  %indvars.iv8.i = phi i64 [ 0, %entry ], [ %indvars.iv.next9.i, %for.cond4.preheader.i ]
  %arrayidx10.i = getelementptr inbounds [4 x [2 x i128]], ptr %l_4774.i, i64 0, i64 %indvars.iv8.i, i64 0
  store i128 8721036757475490113, ptr %arrayidx10.i, align 8
  %arrayidx10.i.c = getelementptr inbounds [4 x [2 x i128]], ptr %l_4774.i, i64 0, i64 %indvars.iv8.i, i64 1
  store i128 8721036757475490113, ptr %arrayidx10.i.c, align 8
  %indvars.iv.next9.i = add nuw nsw i64 %indvars.iv8.i, 1
  %exitcond.not.i = icmp eq i64 %indvars.iv.next9.i, 4
  br i1 %exitcond.not.i, label %func_1.exit, label %for.cond4.preheader.i

func_1.exit:                                      ; preds = %for.cond4.preheader.i
  %arrayidx195.i = getelementptr inbounds [4 x [2 x i128]], ptr %l_4774.i, i64 0, i64 1
  %0 = load i128, ptr %arrayidx195.i, align 8
  %cmp200.i = icmp ne i128 %0, 0
  %conv202.i = zext i1 %cmp200.i to i64
  %call203.i = tail call i64 @safe_sub_func_int64_t_s_s(i64 noundef %conv202.i, i64 noundef 9139899272418802852)
  call void @llvm.lifetime.end.p0(ptr nonnull %l_4774.i)
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %func_1.exit
  br label %for.cond
}

declare void @llvm.lifetime.start.p0(ptr nocapture)
declare void @llvm.lifetime.end.p0(ptr nocapture)
declare dso_local i64 @safe_sub_func_int64_t_s_s(i64, i64)
