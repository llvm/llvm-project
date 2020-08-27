; RUN: opt -loop-vectorize -hexagon-autohvx=1 -force-vector-width=64 -prefer-predicate-over-epilogue=predicate-dont-vectorize -S %s | FileCheck %s

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

; Test for PR45572.

; Check that interleave groups and decisions based on them are correctly
; invalidated with tail-folding on platforms where masked interleaved accesses
; are disabled.

; Make sure a vector body has been created, 64 element vectors are used and a block predicate has been computed.
; Also make sure the loads are not widened.

; CHECK-LABEL: @test1
; CHECK: vector.body:
; CHECK: icmp ule <64 x i32> %vec.ind
; CHECK-NOT: load <{{.*}} x i32>


define void @test1(i32* %arg, i32 %N) #0 {
entry:
  %tmp = alloca i8
  br label %loop

loop:                                              ; preds = %bb2, %bb
  %iv = phi i32 [ %iv.next, %loop], [ 0, %entry ]
  %idx.mul = mul nuw nsw i32 %iv, 7
  %idx.start = add nuw nsw i32 %idx.mul, 1
  %tmp6 = getelementptr inbounds i32, i32* %arg, i32 %idx.start
  %tmp7 = load i32, i32* %tmp6, align 4
  %tmp8 = add nuw nsw i32 %idx.start, 1
  %tmp9 = getelementptr inbounds i32, i32* %arg, i32 %tmp8
  %tmp10 = load i32, i32* %tmp9, align 4
  %tmp11 = add nuw nsw i32 %idx.start, 2
  %tmp12 = getelementptr inbounds i32, i32* %arg, i32 %tmp11
  %tmp13 = load i32, i32* %tmp12, align 4
  %tmp14 = add nuw nsw i32 %idx.start, 3
  %tmp15 = getelementptr inbounds i32, i32* %arg, i32 %tmp14
  %tmp16 = load i32, i32* %tmp15, align 4
  %tmp18 = add nuw nsw i32 %idx.start, 4
  %tmp19 = getelementptr inbounds i32, i32* %arg, i32 %tmp18
  %tmp20 = load i32, i32* %tmp19, align 4
  %tmp21 = add nuw nsw i32 %idx.start, 5
  %tmp22 = getelementptr inbounds i32, i32* %arg, i32 %tmp21
  %tmp23 = load i32, i32* %tmp22, align 4
  %tmp25 = add nuw nsw i32 %idx.start, 6
  %tmp26 = getelementptr inbounds i32, i32* %arg, i32 %tmp25
  %tmp27 = load i32, i32* %tmp26, align 4
  store i8 0, i8* %tmp, align 1
  %iv.next= add nuw nsw i32 %iv, 1
  %exit.cond = icmp eq i32 %iv.next, %N
  br i1 %exit.cond, label %exit, label %loop

exit:                                             ; preds = %loop
  ret void
}

; The loop below only requires tail folding due to interleave groups with gaps.
; Make sure the loads are not widened.

; CHECK-LABEL: @test2
; CHECK: vector.body:
; CHECK-NOT: load <{{.*}} x i32>
define void @test2(i32* %arg) #1 {
entry:
  %tmp = alloca i8
  br label %loop

loop:                                              ; preds = %bb2, %bb
  %iv = phi i32 [ %iv.next, %loop], [ 0, %entry ]
  %idx.start = mul nuw nsw i32 %iv, 5
  %tmp6 = getelementptr inbounds i32, i32* %arg, i32 %idx.start
  %tmp7 = load i32, i32* %tmp6, align 4
  %tmp8 = add nuw nsw i32 %idx.start, 1
  %tmp9 = getelementptr inbounds i32, i32* %arg, i32 %tmp8
  %tmp10 = load i32, i32* %tmp9, align 4
  %tmp11 = add nuw nsw i32 %idx.start, 2
  %tmp12 = getelementptr inbounds i32, i32* %arg, i32 %tmp11
  %tmp13 = load i32, i32* %tmp12, align 4
  %tmp14 = add nuw nsw i32 %idx.start, 3
  %tmp15 = getelementptr inbounds i32, i32* %arg, i32 %tmp14
  %tmp16 = load i32, i32* %tmp15, align 4
  store i8 0, i8* %tmp, align 1
  %iv.next= add nuw nsw i32 %iv, 1
  %exit.cond = icmp eq i32 %iv.next, 128
  br i1 %exit.cond, label %exit, label %loop

exit:                                             ; preds = %loop
  ret void
}


attributes #0 = { "target-features"="+hvx,+hvx-length128b" }
attributes #1 = { optsize "target-features"="+hvx,+hvx-length128b" }
