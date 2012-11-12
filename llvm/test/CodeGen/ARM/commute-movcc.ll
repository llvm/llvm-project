; RUN: llc -mtriple=thumbv7-apple-ios -disable-code-place < %s | FileCheck %s
; RUN: llc -mtriple=armv7-apple-ios   -disable-code-place < %s | FileCheck %s

; LLVM IR optimizers canonicalize icmp+select this way.
; Make sure that TwoAddressInstructionPass can commute the corresponding
; MOVCC instructions to avoid excessive copies in one of the if blocks.
;
; CHECK: %if.then
; CHECK-NOT: mov
; CHECK: movlo
; CHECK: movlo
; CHECK-NOT: mov

; CHECK: %if.else
; CHECK-NOT: mov
; CHECK: movls
; CHECK: movls
; CHECK-NOT: mov

; This is really an LSR test: Make sure that cmp is using the incremented
; induction variable.
; CHECK: %if.end8
; CHECK: add{{(s|\.w)?}} [[IV:r[0-9]+]], {{.*}}#1
; CHECK: cmp [[IV]], #

define i32 @f(i32* nocapture %a, i32 %Pref) nounwind ssp {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %if.end8
  %i.012 = phi i32 [ 0, %entry ], [ %inc, %if.end8 ]
  %BestCost.011 = phi i32 [ -1, %entry ], [ %BestCost.1, %if.end8 ]
  %BestIdx.010 = phi i32 [ 0, %entry ], [ %BestIdx.1, %if.end8 ]
  %arrayidx = getelementptr inbounds i32* %a, i32 %i.012
  %0 = load i32* %arrayidx, align 4, !tbaa !0
  %mul = mul i32 %0, %0
  %sub = add nsw i32 %i.012, -5
  %cmp2 = icmp eq i32 %sub, %Pref
  br i1 %cmp2, label %if.else, label %if.then

if.then:                                          ; preds = %for.body
  %cmp3 = icmp ult i32 %mul, %BestCost.011
  %i.0.BestIdx.0 = select i1 %cmp3, i32 %i.012, i32 %BestIdx.010
  %mul.BestCost.0 = select i1 %cmp3, i32 %mul, i32 %BestCost.011
  br label %if.end8

if.else:                                          ; preds = %for.body
  %cmp5 = icmp ugt i32 %mul, %BestCost.011
  %BestIdx.0.i.0 = select i1 %cmp5, i32 %BestIdx.010, i32 %i.012
  %BestCost.0.mul = select i1 %cmp5, i32 %BestCost.011, i32 %mul
  br label %if.end8

if.end8:                                          ; preds = %if.else, %if.then
  %BestIdx.1 = phi i32 [ %i.0.BestIdx.0, %if.then ], [ %BestIdx.0.i.0, %if.else ]
  %BestCost.1 = phi i32 [ %mul.BestCost.0, %if.then ], [ %BestCost.0.mul, %if.else ]
  store i32 %mul, i32* %arrayidx, align 4, !tbaa !0
  %inc = add i32 %i.012, 1
  %cmp = icmp eq i32 %inc, 11
  br i1 %cmp, label %for.end, label %for.body

for.end:                                          ; preds = %if.end8
  ret i32 %BestIdx.1
}

!0 = metadata !{metadata !"int", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA", null}
