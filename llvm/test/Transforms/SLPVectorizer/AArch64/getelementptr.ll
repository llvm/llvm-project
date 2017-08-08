; RUN: opt -S -slp-vectorizer -slp-threshold=-18 -dce -instcombine -pass-remarks-output=%t < %s | FileCheck %s
; RUN: cat %t | FileCheck -check-prefix=YAML %s

target datalayout = "e-m:e-i32:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

; These tests check that we remove from consideration pairs of seed
; getelementptrs when they are known to have a constant difference. Such pairs
; are likely not good candidates for vectorization since one can be computed
; from the other. We use an unprofitable threshold to force vectorization.
;
; int getelementptr(int *g, int n, int w, int x, int y, int z) {
;   int sum = 0;
;   for (int i = 0; i < n ; ++i) {
;     sum += g[2*i + w]; sum += g[2*i + x];
;     sum += g[2*i + y]; sum += g[2*i + z];
;   }
;   return sum;
; }
;

; CHECK-LABEL: @getelementptr_4x32
;
; CHECK: [[A:%[a-zA-Z0-9.]+]] = add nsw <4 x i32>
; CHECK: [[X:%[a-zA-Z0-9.]+]] = extractelement <4 x i32> [[A]]
; CHECK: sext i32 [[X]] to i64

; YAML:      Pass:            slp-vectorizer
; YAML-NEXT: Name:            VectorizedList
; YAML-NEXT: Function:        getelementptr_4x32
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'SLP vectorized with cost '
; YAML-NEXT:   - Cost:            '11'
; YAML-NEXT:   - String:          ' and with tree size '
; YAML-NEXT:   - TreeSize:        '5'

; YAML:      Pass:            slp-vectorizer
; YAML-NEXT: Name:            VectorizedList
; YAML-NEXT: Function:        getelementptr_4x32
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'SLP vectorized with cost '
; YAML-NEXT:   - Cost:            '16'
; YAML-NEXT:   - String:          ' and with tree size '
; YAML-NEXT:   - TreeSize:        '3'

define i32 @getelementptr_4x32(i32* nocapture readonly %g, i32 %n, i32 %x, i32 %y, i32 %z) {
entry:
  %cmp31 = icmp sgt i32 %n, 0
  br i1 %cmp31, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  br label %for.body

for.cond.cleanup.loopexit:
  br label %for.cond.cleanup

for.cond.cleanup:
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add16, %for.cond.cleanup.loopexit ]
  ret i32 %sum.0.lcssa

for.body:
  %indvars.iv = phi i32 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %sum.032 = phi i32 [ 0, %for.body.preheader ], [ %add16, %for.body ]
  %t4 = shl nsw i32 %indvars.iv, 1
  %t5 = add nsw i32 %t4, 0
  %arrayidx = getelementptr inbounds i32, i32* %g, i32 %t5
  %t6 = load i32, i32* %arrayidx, align 4
  %add1 = add nsw i32 %t6, %sum.032
  %t7 = add nsw i32 %t4, %x
  %arrayidx5 = getelementptr inbounds i32, i32* %g, i32 %t7
  %t8 = load i32, i32* %arrayidx5, align 4
  %add6 = add nsw i32 %add1, %t8
  %t9 = add nsw i32 %t4, %y
  %arrayidx10 = getelementptr inbounds i32, i32* %g, i32 %t9
  %t10 = load i32, i32* %arrayidx10, align 4
  %add11 = add nsw i32 %add6, %t10
  %t11 = add nsw i32 %t4, %z
  %arrayidx15 = getelementptr inbounds i32, i32* %g, i32 %t11
  %t12 = load i32, i32* %arrayidx15, align 4
  %add16 = add nsw i32 %add11, %t12
  %indvars.iv.next = add nuw nsw i32 %indvars.iv, 1
  %exitcond = icmp eq i32 %indvars.iv.next , %n
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body
}

; CHECK-LABEL: @getelementptr_2x32
;
; CHECK: [[A:%[a-zA-Z0-9.]+]] = add nsw <2 x i32>
; CHECK: [[X:%[a-zA-Z0-9.]+]] = extractelement <2 x i32> [[A]]
; CHECK: sext i32 [[X]] to i64

; YAML:      Pass:            slp-vectorizer
; YAML-NEXT: Name:            VectorizedList
; YAML-NEXT: Function:        getelementptr_2x32
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'SLP vectorized with cost '
; YAML-NEXT:   - Cost:            '11'
; YAML-NEXT:   - String:          ' and with tree size '
; YAML-NEXT:   - TreeSize:        '5'

; YAML:      Pass:            slp-vectorizer
; YAML-NEXT: Name:            VectorizedList
; YAML-NEXT: Function:        getelementptr_2x32
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'SLP vectorized with cost '
; YAML-NEXT:   - Cost:            '6'
; YAML-NEXT:   - String:          ' and with tree size '
; YAML-NEXT:   - TreeSize:        '3'

define i32 @getelementptr_2x32(i32* nocapture readonly %g, i32 %n, i32 %x, i32 %y, i32 %z) {
entry:
  %cmp31 = icmp sgt i32 %n, 0
  br i1 %cmp31, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  br label %for.body

for.cond.cleanup.loopexit:
  br label %for.cond.cleanup

for.cond.cleanup:
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add16, %for.cond.cleanup.loopexit ]
  ret i32 %sum.0.lcssa

for.body:
  %indvars.iv = phi i32 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %sum.032 = phi i32 [ 0, %for.body.preheader ], [ %add16, %for.body ]
  %t4 = shl nsw i32 %indvars.iv, 1
  %t5 = add nsw i32 %t4, 0
  %arrayidx = getelementptr inbounds i32, i32* %g, i32 %t5
  %t6 = load i32, i32* %arrayidx, align 4
  %add1 = add nsw i32 %t6, %sum.032
  %t7 = add nsw i32 %t4, 1
  %arrayidx5 = getelementptr inbounds i32, i32* %g, i32 %t7
  %t8 = load i32, i32* %arrayidx5, align 4
  %add6 = add nsw i32 %add1, %t8
  %t9 = add nsw i32 %t4, %y
  %arrayidx10 = getelementptr inbounds i32, i32* %g, i32 %t9
  %t10 = load i32, i32* %arrayidx10, align 4
  %add11 = add nsw i32 %add6, %t10
  %t11 = add nsw i32 %t4, %z
  %arrayidx15 = getelementptr inbounds i32, i32* %g, i32 %t11
  %t12 = load i32, i32* %arrayidx15, align 4
  %add16 = add nsw i32 %add11, %t12
  %indvars.iv.next = add nuw nsw i32 %indvars.iv, 1
  %exitcond = icmp eq i32 %indvars.iv.next , %n
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body
}
