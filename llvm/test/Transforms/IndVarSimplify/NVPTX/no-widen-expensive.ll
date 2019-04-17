; RUN: opt < %s -indvars -S | FileCheck %s

target triple = "nvptx64-unknown-unknown"

; For the nvptx64 architecture, the cost of an arithmetic instruction on a
; 64-bit integer is twice as expensive as that on a 32-bit integer, because the
; hardware needs to simulate a 64-bit integer using two 32-bit integers.
; Therefore, in this particular architecture, we should not widen induction
; variables to 64-bit integers even though i64 is a legal type in the 64-bit
; PTX ISA.

define void @indvar_32_bit(i32 %n, i32* nocapture %output) {
; CHECK-LABEL: @indvar_32_bit
entry:
  %cmp5 = icmp sgt i32 %n, 0
  br i1 %cmp5, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.06 = phi i32 [ 0, %for.body.preheader ], [ %add, %for.body ]
; CHECK: phi i32
  %mul = mul nsw i32 %i.06, %i.06
  %0 = sext i32 %i.06 to i64
  %arrayidx = getelementptr inbounds i32, i32* %output, i64 %0
  store i32 %mul, i32* %arrayidx, align 4
  %add = add nsw i32 %i.06, 3
  %cmp = icmp slt i32 %add, %n
  br i1 %cmp, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}
