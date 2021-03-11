; RUN: opt < %s -loop-vectorize -mtriple=riscv64 -mattr=+experimental-v -riscv-v-vector-bits-min=128 -S | FileCheck %s
; RUN: opt < %s -loop-vectorize -mtriple=riscv32 -mattr=+experimental-v -riscv-v-vector-bits-min=128 -S | FileCheck %s

; Function Attrs: nounwind
define i32* @array_add(i32* noalias nocapture readonly %a, i32* noalias nocapture readonly %b, i32* %c, i32 %size) {
;CHECK-LABEL: array_add
;CHECK: load <4 x i32>
;CHECK: load <4 x i32>
;CHECK: add nsw <4 x i32>
;CHECK: store <4 x i32>
;CHECK: ret
entry:
  %cmp10 = icmp sgt i32 %size, 0
  br i1 %cmp10, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %1 = load i32, i32* %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %arrayidx4 = getelementptr inbounds i32, i32* %c, i64 %indvars.iv
  store i32 %add, i32* %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %size
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret i32* %c
}
