; RUN: opt < %s -passes=loop-vectorize,dce,instcombine -force-vector-width=4 -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; tests skipping iterations within a VF through break/continue/gotos.

; The main difficulty in vectorizing these loops in test1,test2 and test3 is
; safely speculating that the widened load of A[i] should not fault if the
; scalarized loop does not fault. For example, the
; original load in the scalar loop may not fault, but the last iteration of the
; vectorized load can fault (if it crosses a page boudary for example).
; This last vector iteration is where *one* of the
; scalar iterations lead to the early exit.

; int test(int *A, int Length) {
;   for (int i = 0; i < Length; i++) {
;     if (A[i] > 10.0) goto end;
;     A[i] = 0;
;   }
; end:
;   return 0;
; }
; CHECK-LABEL: test1(
; CHECK-NOT: <4 x i32>
define i32 @test1(i32* nocapture %A, i32 %Length) {
entry:
  %cmp8 = icmp sgt i32 %Length, 0
  br i1 %cmp8, label %for.body.preheader, label %end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %if.else
  %indvars.iv = phi i64 [ %indvars.iv.next, %if.else ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4, !tbaa !15
  %cmp1 = icmp sgt i32 %0, 10
  br i1 %cmp1, label %end.loopexit, label %if.else

if.else:                                          ; preds = %for.body
  store i32 0, i32* %arrayidx, align 4, !tbaa !15
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %1 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %1, %Length
  br i1 %cmp, label %for.body, label %end.loopexit

end.loopexit:                                     ; preds = %if.else, %for.body
  br label %end

end:                                              ; preds = %end.loopexit, %entry
  ret i32 0
}

; We don't use anything from within the loop at the early exit path
; so we do not need to know which iteration caused the early exit path.
; bool test2(int *A, int Length, int K) {
;   for (int i = 0; i < Length; i++) {
;     if (A[i] == K) return true;
;   }
;   return false;
; }
; TODO: Today we do not vectorize this, but we could teach the vectorizer, once
; the hard part of proving/speculating A[i:VF - 1] loads does not fault is handled by the
; compiler/hardware.

; CHECK-LABEL: test2(
; CHECK-NOT: <4 x i32>
define i32 @test2(i32* nocapture %A, i32 %Length, i32 %K) {
entry:
  %cmp8 = icmp sgt i32 %Length, 0
  br i1 %cmp8, label %for.body.preheader, label %end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %if.else
  %indvars.iv = phi i64 [ %indvars.iv.next, %if.else ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %ld = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp eq i32 %ld, %K
  br i1 %cmp1, label %end.loopexit, label %if.else

if.else:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %trunc = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %trunc, %Length
  br i1 %cmp, label %for.body, label %end.loopexit

end.loopexit:                                     ; preds = %if.else, %for.body
  %result.lcssa = phi i32 [ 1, %for.body ], [ 0, %if.else ]
  br label %end

end:                                              ; preds = %end.loopexit, %entry
  %result = phi i32 [ %result.lcssa, %end.loopexit ], [ 0, %entry ]
  ret i32 %result
}

; We use the IV in the early exit
; so we need to know which iteration caused the early exit path.
; int test3(int *A, int Length, int K) {
;   for (int i = 0; i < Length; i++) {
;     if (A[i] == K) return i;
;   }
;   return -1;
; }
; TODO: Today we do not vectorize this, but we could teach the vectorizer (once
; we handle the speculation safety of the widened load).
; CHECK-LABEL: test3(
; CHECK-NOT: <4 x i32>
define i32 @test3(i32* nocapture %A, i32 %Length, i32 %K) {
entry:
  %cmp8 = icmp sgt i32 %Length, 0
  br i1 %cmp8, label %for.body.preheader, label %end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %if.else
  %indvars.iv = phi i64 [ %indvars.iv.next, %if.else ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %ld = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp eq i32 %ld, %K
  br i1 %cmp1, label %end.loopexit, label %if.else

if.else:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %trunc = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %trunc, %Length
  br i1 %cmp, label %for.body, label %end.loopexit

end.loopexit:                                     ; preds = %if.else, %for.body
  %result.lcssa = phi i64 [ %indvars.iv, %for.body ], [ -1, %if.else ]
  %res.trunc = trunc i64 %result.lcssa to i32
  br label %end

end:                                              ; preds = %end.loopexit, %entry
  %result = phi i32 [ %res.trunc, %end.loopexit ], [ -1, %entry ]
  ret i32 %result
}

; bool test4(int *A, int Length, int K, int J) {
;   for (int i = 0; i < Length; i++) {
;     if (A[i] == K) continue;
;     A[i] = J;
;   }
; }
; For this test, we vectorize and generate predicated stores to A[i].
; CHECK-LABEL: test4(
; CHECK: <4 x i32>
define void @test4(i32* nocapture %A, i32 %Length, i32 %K, i32 %J) {
entry:
  %cmp8 = icmp sgt i32 %Length, 0
  br i1 %cmp8, label %for.body.preheader, label %end.loopexit

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %if.else
  %indvars.iv = phi i64 [ %indvars.iv.next, %latch ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %trunc = trunc i64 %indvars.iv.next to i32
  %ld = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp eq i32 %ld, %K
  br i1 %cmp1, label %latch, label %if.else

if.else:
  store i32 %J, i32* %arrayidx, align 4
  br label %latch

latch:                                          ; preds = %for.body
  %cmp = icmp slt i32 %trunc, %Length
  br i1 %cmp, label %for.body, label %end.loopexit

end.loopexit:                                     ; preds = %if.else, %for.body
  ret void
}
!15 = !{!16, !16, i64 0}
!16 = !{!"int", !17, i64 0}
!17 = !{!"omnipotent char", !18, i64 0}
!18 = !{!"Simple C/C++ TBAA"}
