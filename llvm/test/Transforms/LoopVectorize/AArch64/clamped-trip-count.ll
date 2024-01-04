; RUN: opt -S < %s -passes=loop-vectorize -mtriple aarch64-linux-gnu -mattr=+sve 2>&1 | FileCheck %s

define void @clamped_tc_8(ptr nocapture %dst, i32 %n, i64 %val){
; CHECK-LABEL: define void @clamped_tc_8
; CHECK: call void @llvm.masked.store.nxv8i8.p0(<vscale x 8 x i8> {{.*}}, ptr {{.*}}, i32 1, <vscale x 8 x i1> {{.*}})
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %p_out_tail.09 = phi ptr [ %dst, %entry ], [ %incdec.ptr, %for.body ]
  %0 = shl nuw nsw i64 %indvars.iv, 3
  %shr3 = lshr i64 %val, %0
  %conv4 = trunc i64 %shr3 to i8
  store i8 %conv4, ptr %p_out_tail.09, align 1
  %incdec.ptr = getelementptr inbounds i8, ptr %p_out_tail.09, i64 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 8
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

define void @clamped_tc_max_8(ptr nocapture %dst, i32 %n, i64 %val){
; CHECK-LABEL: define void @clamped_tc_max_8(
; CHECK: call void @llvm.masked.store.nxv8i8.p0(<vscale x 8 x i8> {{.*}}, ptr {{.*}}, i32 1, <vscale x 8 x i1> {{.*}})

entry:
  %rem = and i32 %n, 63
  %cmp8.not = icmp eq i32 %rem, 0
  br i1 %cmp8.not, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %add = add nuw nsw i32 %rem, 7
  %shr = lshr i32 %add, 3
  %wide.trip.count = zext i32 %shr to i64
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %p_out_tail.09 = phi ptr [ %dst, %for.body.preheader ], [ %incdec.ptr, %for.body ]
  %0 = shl nuw nsw i64 %indvars.iv, 3
  %shr3 = lshr i64 %val, %0
  %conv4 = trunc i64 %shr3 to i8
  store i8 %conv4, ptr %p_out_tail.09, align 1
  %incdec.ptr = getelementptr inbounds i8, ptr %p_out_tail.09, i64 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}
