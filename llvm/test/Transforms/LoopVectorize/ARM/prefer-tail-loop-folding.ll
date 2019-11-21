; RUN: opt -mtriple=thumbv8.1m.main-arm-eabihf \
; RUN:   -disable-mve-tail-predication=false -loop-vectorize -S < %s | \
; RUN:   FileCheck %s -check-prefixes=CHECK,NO-FOLDING

; RUN: opt -mtriple=thumbv8.1m.main-arm-eabihf -mattr=-mve \
; RUN:   -disable-mve-tail-predication=false -loop-vectorize \
; RUN:   -enable-arm-maskedldst=true -S < %s | \
; RUN:   FileCheck %s -check-prefixes=CHECK,NO-FOLDING

; RUN: opt -mtriple=thumbv8.1m.main-arm-eabihf -mattr=+mve \
; RUN:   -disable-mve-tail-predication=false -loop-vectorize \
; RUN:   -enable-arm-maskedldst=false -S < %s | \
; RUN:   FileCheck %s -check-prefixes=CHECK,NO-FOLDING

; RUN: opt -mtriple=thumbv8.1m.main-arm-eabihf -mattr=+mve \
; RUN:   -disable-mve-tail-predication=true -loop-vectorize \
; RUN:   -enable-arm-maskedldst=true -S < %s | \
; RUN:   FileCheck %s -check-prefixes=CHECK,NO-FOLDING

; Disabling the low-overhead branch extension will make
; 'isHardwareLoopProfitable' return false, so that we test avoiding folding for
; these cases.
; RUN: opt -mtriple=thumbv8.1m.main-arm-eabihf -mattr=+mve,-lob \
; RUN:   -disable-mve-tail-predication=false -loop-vectorize \
; RUN:   -enable-arm-maskedldst=true -S < %s | \
; RUN:   FileCheck %s -check-prefixes=CHECK,NO-FOLDING

; RUN: opt -mtriple=thumbv8.1m.main-arm-eabihf -mattr=+mve.fp \
; RUN:   -disable-mve-tail-predication=false -loop-vectorize \
; RUN:   -enable-arm-maskedldst=true -S < %s | \
; RUN:   FileCheck %s -check-prefixes=CHECK,PREFER-FOLDING

; RUN: opt -mtriple=thumbv8.1m.main-arm-eabihf -mattr=+mve.fp \
; RUN:   -prefer-predicate-over-epilog=false \
; RUN:   -disable-mve-tail-predication=false -loop-vectorize \
; RUN:   -enable-arm-maskedldst=true -S < %s | \
; RUN:   FileCheck %s -check-prefixes=CHECK,NO-FOLDING

; RUN: opt -mtriple=thumbv8.1m.main-arm-eabihf -mattr=+mve.fp \
; RUN:   -prefer-predicate-over-epilog=true \
; RUN:   -disable-mve-tail-predication=false -loop-vectorize \
; RUN:   -enable-arm-maskedldst=true -S < %s | \
; RUN:   FileCheck %s -check-prefixes=CHECK,FOLDING-OPT

define void @prefer_folding(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C) #0 {
; CHECK-LABEL:    prefer_folding(
; PREFER-FOLDING: vector.body:
; PREFER-FOLDING: call <4 x i32> @llvm.masked.load.v4i32.p0v4i32
; PREFER-FOLDING: call <4 x i32> @llvm.masked.load.v4i32.p0v4i32
; PREFER-FOLDING: call void @llvm.masked.store.v4i32.p0v4i32
; PREFER-FOLDING: br i1 %{{.*}}, label %{{.*}}, label %vector.body
;
; NO-FOLDING-NOT: call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(
; NO-FOLDING-NOT: call void @llvm.masked.store.v4i32.p0v4i32(
; NO-FOLDING:     br i1 %{{.*}}, label %{{.*}}, label %for.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %i.09
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %C, i32 %i.09
  %1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %1, %0
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i32 %i.09
  store i32 %add, i32* %arrayidx2, align 4
  %add3 = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %add3, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define void @mixed_types(i16* noalias nocapture %A, i16* noalias nocapture readonly %B, i16* noalias nocapture readonly %C, i32* noalias nocapture %D, i32* noalias nocapture readonly %E, i32* noalias nocapture readonly %F) #0 {
; CHECK-LABEL:        mixed_types(
; PREFER-FOLDING:     vector.body:
; PREFER-FOLDING:     call <4 x i16> @llvm.masked.load.v4i16.p0v4i16
; PREFER-FOLDING:     call <4 x i16> @llvm.masked.load.v4i16.p0v4i16
; PREFER-FOLDING:     call void @llvm.masked.store.v4i16.p0v4i16
; PREFER-FOLDING:     call <4 x i32> @llvm.masked.load.v4i32.p0v4i32
; PREFER-FOLDING:     call <4 x i32> @llvm.masked.load.v4i32.p0v4i32
; PREFER-FOLDING:     call void @llvm.masked.store.v4i32.p0v4i32
; PREFER-FOLDING:     br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.018 = phi i32 [ 0, %entry ], [ %add9, %for.body ]
  %arrayidx = getelementptr inbounds i16, i16* %B, i32 %i.018
  %0 = load i16, i16* %arrayidx, align 2
  %arrayidx1 = getelementptr inbounds i16, i16* %C, i32 %i.018
  %1 = load i16, i16* %arrayidx1, align 2
  %add = add i16 %1, %0
  %arrayidx4 = getelementptr inbounds i16, i16* %A, i32 %i.018
  store i16 %add, i16* %arrayidx4, align 2
  %arrayidx5 = getelementptr inbounds i32, i32* %E, i32 %i.018
  %2 = load i32, i32* %arrayidx5, align 4
  %arrayidx6 = getelementptr inbounds i32, i32* %F, i32 %i.018
  %3 = load i32, i32* %arrayidx6, align 4
  %add7 = add nsw i32 %3, %2
  %arrayidx8 = getelementptr inbounds i32, i32* %D, i32 %i.018
  store i32 %add7, i32* %arrayidx8, align 4
  %add9 = add nuw nsw i32 %i.018, 1
  %exitcond = icmp eq i32 %add9, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define void @unsupported_i64_type(i64* noalias nocapture %A, i64* noalias nocapture readonly %B, i64* noalias nocapture readonly %C) #0 {
; CHECK-LABEL:        unsupported_i64_type(
; PREFER-FOLDING-NOT: vector.body:
; PREFER-FOLDING-NOT: llvm.masked.load
; PREFER-FOLDING-NOT: llvm.masked.store
; PREFER-FOLDING:     for.body:
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds i64, i64* %B, i32 %i.09
  %0 = load i64, i64* %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds i64, i64* %C, i32 %i.09
  %1 = load i64, i64* %arrayidx1, align 8
  %add = add nsw i64 %1, %0
  %arrayidx2 = getelementptr inbounds i64, i64* %A, i32 %i.09
  store i64 %add, i64* %arrayidx2, align 8
  %add3 = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %add3, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define void @zero_extending_load_allowed(i32* noalias nocapture %A, i8* noalias nocapture readonly %B, i32* noalias nocapture readonly %C) #0 {
; CHECK-LABEL:    zero_extending_load_allowed(
; PREFER-FOLDING: vector.body:
; PREFER-FOLDING: call <4 x i8> @llvm.masked.load.v4i8.p0v4i8
; PREFER-FOLDING: call <4 x i32> @llvm.masked.load.v4i32.p0v4i32
; PREFER-FOLDING: call void @llvm.masked.store.v4i32.p0v4i32
; PREFER-FOLDING: br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds i8, i8* %B, i32 %i.09
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %arrayidx1 = getelementptr inbounds i32, i32* %C, i32 %i.09
  %1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %1, %conv
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i32 %i.09
  store i32 %add, i32* %arrayidx2, align 4
  %add3 = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %add3, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define void @sign_extending_load_allowed(i32* noalias nocapture %A, i8* noalias nocapture readonly %B, i32* noalias nocapture readonly %C) #0 {
; CHECK-LABEL:    sign_extending_load_allowed(
; PREFER-FOLDING: vector.body:
; PREFER-FOLDING: call <4 x i8> @llvm.masked.load.v4i8.p0v4i8
; PREFER-FOLDING: call <4 x i32> @llvm.masked.load.v4i32.p0v4i32
; PREFER-FOLDING: call void @llvm.masked.store.v4i32.p0v4i32
; PREFER-FOLDING: br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds i8, i8* %B, i32 %i.09
  %0 = load i8, i8* %arrayidx, align 1
  %conv = sext i8 %0 to i32
  %arrayidx1 = getelementptr inbounds i32, i32* %C, i32 %i.09
  %1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %1, %conv
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i32 %i.09
  store i32 %add, i32* %arrayidx2, align 4
  %add3 = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %add3, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define void @narrowing_load_not_allowed(i8* noalias nocapture %A, i8* noalias nocapture readonly %B, i16* noalias nocapture readonly %C) #0 {
; CHECK-LABEL:        narrowing_load_not_allowed(
; PREFER-FOLDING:     vector.body:
; PREFER-FOLDING-NOT: llvm.masked.load
; PREFER-FOLDING-NOT: llvm.masked.store
; PREFER-FOLDING:     br i1 %{{.*}}, label %{{.*}}, label %vector.body

; FOLDING-OPT:        vector.body:
; FOLDING-OPT         call <8 x i16> @llvm.masked.load.v8i16.p0v8i16
; FOLDING-OPT         call <8 x i8> @llvm.masked.load.v8i8.p0v8i8
; FOLDING-OPT         call void @llvm.masked.store.v8i8.p0v8i8
; FOLDING-OPT:        br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %for.body, %entry
  %i.012 = phi i32 [ 0, %entry ], [ %add6, %for.body ]
  %arrayidx = getelementptr inbounds i16, i16* %C, i32 %i.012
  %0 = load i16, i16* %arrayidx, align 2
  %arrayidx1 = getelementptr inbounds i8, i8* %B, i32 %i.012
  %1 = load i8, i8* %arrayidx1, align 1
  %conv3 = trunc i16 %0 to i8
  %add = add i8 %1, %conv3
  %arrayidx5 = getelementptr inbounds i8, i8* %A, i32 %i.012
  store i8 %add, i8* %arrayidx5, align 1
  %add6 = add nuw nsw i32 %i.012, 1
  %exitcond = icmp eq i32 %add6, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define void @narrowing_store_allowed(i8* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C) #0 {
; CHECK-LABEL:    narrowing_store_allowed(
; PREFER-FOLDING: call void @llvm.masked.store.v4i8.p0v4i8
; PREFER-FOLDING: br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %i.09
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %C, i32 %i.09
  %1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %1, %0
  %conv = trunc i32 %add to i8
  %arrayidx2 = getelementptr inbounds i8, i8* %A, i32 %i.09
  store i8 %conv, i8* %arrayidx2, align 1
  %add3 = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %add3, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; This is a trunc not connected to a store, so we don't allow this.
; TODO: this is conservative, because the trunc is only used in the
; loop control statements, and thus not affecting element sizes, so
; we could allow this case.
define void @trunc_not_allowed(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C) #0 {
; CHECK-LABEL:        trunc_not_allowed(
; PREFER-FOLDING:     vector.body:
; PREFER-FOLDING-NOT: llvm.masked.load
; PREFER-FOLDING-NOT: llvm.masked.store
; PREFER-FOLDING:     br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %i.09
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %C, i32 %i.09
  %1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %1, %0
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i32 %i.09
  store i32 %add, i32* %arrayidx2, align 4
  %add3 = add nuw nsw i32 %i.09, 1

  %add.iv = trunc i32 %add3 to i16

  %exitcond = icmp eq i16 %add.iv, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define void @trunc_not_allowed_different_vec_elemns(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C, i16* noalias nocapture %D) #0 {
; CHECK-LABEL:        trunc_not_allowed_different_vec_elemns(
; PREFER-FOLDING:     vector.body:
; PREFER-FOLDING-NOT: llvm.masked.load
; PREFER-FOLDING-NOT: llvm.masked.store
; PREFER-FOLDING:     br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.021 = phi i32 [ 0, %entry ], [ %add9, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %i.021
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %C, i32 %i.021
  %1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %1, %0
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i32 %i.021
  store i32 %add, i32* %arrayidx2, align 4
  %add.tr = trunc i32 %add to i16
  %conv7 = shl i16 %add.tr, 1
  %arrayidx8 = getelementptr inbounds i16, i16* %D, i32 %i.021
  store i16 %conv7, i16* %arrayidx8, align 2
  %add9 = add nuw nsw i32 %i.021, 1
  %exitcond = icmp eq i32 %add9, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}


@tab = common global [32 x i8] zeroinitializer, align 1

define i32 @icmp_not_allowed() #0 {
; CHECK-LABEL:        icmp_not_allowed(
; PREFER-FOLDING:     vector.body:
; PREFER-FOLDING-NOT: llvm.masked.load
; PREFER-FOLDING-NOT: llvm.masked.store
; PREFER-FOLDING:     br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.body:
  %i.08 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds [32 x i8], [32 x i8]* @tab, i32 0, i32 %i.08
  %0 = load i8, i8* %arrayidx, align 1
  %cmp1 = icmp eq i8 %0, 0
  %. = select i1 %cmp1, i8 2, i8 1
  store i8 %., i8* %arrayidx, align 1
  %inc = add nsw i32 %i.08, 1
  %exitcond = icmp slt i32 %inc, 1000
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret i32 0
}

@ftab = common global [32 x float] zeroinitializer, align 1

define float @fcmp_not_allowed() #0 {
; CHECK-LABEL:        fcmp_not_allowed(
; PREFER-FOLDING:     vector.body:
; PREFER-FOLDING-NOT: llvm.masked.load
; PREFER-FOLDING-NOT: llvm.masked.store
; PREFER-FOLDING:     br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.body:
  %i.08 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds [32 x float], [32 x float]* @ftab, i32 0, i32 %i.08
  %0 = load float, float* %arrayidx, align 4
  %cmp1 = fcmp oeq float %0, 0.000000e+00
  %. = select i1 %cmp1, float 2.000000e+00, float 1.000000e+00
  store float %., float* %arrayidx, align 4
  %inc = add nsw i32 %i.08, 1
  %exitcond = icmp slt i32 %inc, 999
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret float 0.000000e+00
}

define void @pragma_vect_predicate_disable(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C) #0 {
; CHECK-LABEL:        pragma_vect_predicate_disable(
; PREFER-FOLDING:     vector.body:
; PREFER-FOLDING-NOT: call <4 x i32> @llvm.masked.load.v4i32.p0v4i32
; PREFER-FOLDING-NOT: call <4 x i32> @llvm.masked.load.v4i32.p0v4i32
; PREFER-FOLDING-NOT: call void @llvm.masked.store.v4i32.p0v4i32
; PREFER-FOLDING:     br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %i.09
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %C, i32 %i.09
  %1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %1, %0
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i32 %i.09
  store i32 %add, i32* %arrayidx2, align 4
  %add3 = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %add3, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !llvm.loop !7
}

; Test directions for array indices i and N-1. I.e. check strides 1 and -1, and
; force vectorisation with a loop hint.
define void @strides_different_direction(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C, i32 %N) #0 {
; CHECK-LABEL: strides_different_direction(
; PREFER-FOLDING:     vector.body:
; PREFER-FOLDING-NOT: llvm.masked.load
; PREFER-FOLDING-NOT: llvm.masked.store
; PREFER-FOLDING:     br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %i.09
  %0 = load i32, i32* %arrayidx, align 4
  %sub = sub nsw i32 %N, %i.09
  %arrayidx1 = getelementptr inbounds i32, i32* %C, i32 %sub
  %1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %1, %0
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i32 %i.09
  store i32 %add, i32* %arrayidx2, align 4
  %add3 = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %add3, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !llvm.loop !10
}

define void @stride_4(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C) #0 {
; CHECK-LABEL:        stride_4(
; PREFER-FOLDING:     vector.body:
; PREFER-FOLDING-NOT: llvm.masked.load
; PREFER-FOLDING-NOT: llvm.masked.store
; PREFER-FOLDING:     br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %i.09
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %C, i32 %i.09
  %1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %1, %0
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i32 %i.09
  store i32 %add, i32* %arrayidx2, align 4
  %add3 = add nuw nsw i32 %i.09, 4
  %cmp = icmp ult i32 %add3, 731
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !llvm.loop !5
}

define void @too_many_loop_blocks(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C) #0 {
; CHECK-LABEL:        too_many_loop_blocks(
; PREFER-FOLDING:     vector.body:
; PREFER-FOLDING-NOT: llvm.masked.load
; PREFER-FOLDING-NOT: llvm.masked.store
; PREFER-FOLDING:     br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %loopincr ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %i.09
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %C, i32 %i.09
  %1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %1, %0
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i32 %i.09
  store i32 %add, i32* %arrayidx2, align 4
  br label %loopincr

loopincr:
  %add3 = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %add3, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define dso_local void @half(half* noalias nocapture %A, half* noalias nocapture readonly %B, half* noalias nocapture readonly %C) #0 {
; CHECK-LABEL:    half(
; PREFER-FOLDING: vector.body:
; PREFER-FOLDING: call <8 x half> @llvm.masked.load.v8f16.p0v8f16
; PREFER-FOLDING: call <8 x half> @llvm.masked.load.v8f16.p0v8f16
; PREFER-FOLDING: call void @llvm.masked.store.v8f16.p0v8f16
; PREFER-FOLDING: br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds half, half* %B, i32 %i.09
  %0 = load half, half* %arrayidx, align 2
  %arrayidx1 = getelementptr inbounds half, half* %C, i32 %i.09
  %1 = load half, half* %arrayidx1, align 2
  %add = fadd fast half %1, %0
  %arrayidx2 = getelementptr inbounds half, half* %A, i32 %i.09
  store half %add, half* %arrayidx2, align 2
  %add3 = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %add3, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define void @float(float* noalias nocapture %A, float* noalias nocapture readonly %B, float* noalias nocapture readonly %C) #0 {
; CHECK-LABEL:    float(
; PREFER-FOLDING: vector.body:
; PREFER-FOLDING: call <4 x float> @llvm.masked.load.v4f32.p0v4f32
; PREFER-FOLDING: call <4 x float> @llvm.masked.load.v4f32.p0v4f32
; PREFER-FOLDING: call void @llvm.masked.store.v4f32.p0v4f32
; PREFER-FOLDING: br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %B, i32 %i.09
  %0 = load float, float* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, float* %C, i32 %i.09
  %1 = load float, float* %arrayidx1, align 4
  %add = fadd fast float %1, %0
  %arrayidx2 = getelementptr inbounds float, float* %A, i32 %i.09
  store float %add, float* %arrayidx2, align 4
  %add3 = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %add3, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !llvm.loop !10
}

define void @double(double* noalias nocapture %A, double* noalias nocapture readonly %B, double* noalias nocapture readonly %C) #0 {
; CHECK-LABEL:        double(
; PREFER-FOLDING:     for.body:
; PREFER-FOLDING-NOT: vector.body:
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %B, i32 %i.09
  %0 = load double, double* %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds double, double* %C, i32 %i.09
  %1 = load double, double* %arrayidx1, align 8
  %add = fadd fast double %1, %0
  %arrayidx2 = getelementptr inbounds double, double* %A, i32 %i.09
  store double %add, double* %arrayidx2, align 8
  %add3 = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %add3, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; TODO: this fpext could be allowed, but we don't lower it very efficiently yet,
; so reject this for now.
define void @fpext_allowed(float* noalias nocapture %A, half* noalias nocapture readonly %B, float* noalias nocapture readonly %C) #0 {
; CHECK-LABEL:        fpext_allowed(
; PREFER-FOLDING:     vector.body:
; PREFER-FOLDING-NOT: llvm.masked.load
; PREFER-FOLDING-NOT: llvm.masked.store
; PREFER-FOLDING:     br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds half, half* %B, i32 %i.09
  %0 = load half, half* %arrayidx, align 2
  %conv = fpext half %0 to float
  %arrayidx1 = getelementptr inbounds float, float* %C, i32 %i.09
  %1 = load float, float* %arrayidx1, align 4
  %add = fadd fast float %1, %conv
  %arrayidx2 = getelementptr inbounds float, float* %A, i32 %i.09
  store float %add, float* %arrayidx2, align 4
  %add3 = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %add3, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; TODO: this fptrunc could be allowed, but we don't lower it very efficiently yet,
; so reject this for now.
define void @fptrunc_allowed(half* noalias nocapture %A, float* noalias nocapture readonly %B, float* noalias nocapture readonly %C) #0 {
; CHECK-LABEL:        fptrunc_allowed(
; PREFER-FOLDING:     vector.body:
; PREFER-FOLDING-NOT: llvm.masked.load
; PREFER-FOLDING-NOT: llvm.masked.store
; PREFER-FOLDING:     br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %B, i32 %i.09
  %0 = load float, float* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, float* %C, i32 %i.09
  %1 = load float, float* %arrayidx1, align 4
  %add = fadd fast float %1, %0
  %conv = fptrunc float %add to half
  %arrayidx2 = getelementptr inbounds half, half* %A, i32 %i.09
  store half %conv, half* %arrayidx2, align 2
  %add3 = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %add3, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define void @fptrunc_not_allowed(float* noalias nocapture %A, float* noalias nocapture readonly %B, float* noalias nocapture readonly %C, half* noalias nocapture %D) #0 {
; CHECK-LABEL:        fptrunc_not_allowed(
; PREFER-FOLDING:     vector.body:
; PREFER-FOLDING-NOT: llvm.masked.load
; PREFER-FOLDING-NOT: llvm.masked.store
; PREFER-FOLDING:     br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.017 = phi i32 [ 0, %entry ], [ %add6, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %B, i32 %i.017
  %0 = load float, float* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, float* %C, i32 %i.017
  %1 = load float, float* %arrayidx1, align 4
  %add = fadd fast float %1, %0
  %arrayidx2 = getelementptr inbounds float, float* %A, i32 %i.017
  store float %add, float* %arrayidx2, align 4
  %conv = fptrunc float %add to half
  %factor = fmul fast half %conv, 0xH4000
  %arrayidx5 = getelementptr inbounds half, half* %D, i32 %i.017
  store half %factor, half* %arrayidx5, align 2
  %add6 = add nuw nsw i32 %i.017, 1
  %exitcond = icmp eq i32 %add6, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

attributes #0 = { nofree norecurse nounwind "target-features"="+armv8.1-m.main,+mve.fp" }

!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.vectorize.enable", i1 true}

!7 = distinct !{!7, !8}
!8 = !{!"llvm.loop.vectorize.predicate.enable", i1 false}

!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.vectorize.width", i32 4}
