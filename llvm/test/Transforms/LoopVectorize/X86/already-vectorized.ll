; RUN: opt < %s -disable-loop-unrolling -debug-only=loop-vectorize -passes="default<O3>" -S 2>&1 | FileCheck %s
; RUN: opt < %s -disable-loop-unrolling -debug-only=loop-vectorize -O3 -S 2>&1 | FileCheck %s
; REQUIRES: asserts

; We want to make sure that we don't even try to vectorize loops again
; The vectorizer used to mark the un-vectorized loop only as already vectorized
; thus, trying to vectorize the vectorized loop again

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = external global [255 x i32]

; Function Attrs: nounwind readonly uwtable
define i32 @vect() {
; CHECK: LV: Checking a loop in 'vect'
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
; We need to make sure we did vectorize the loop
; CHECK: LV: Found a loop: for.body
; CHECK: LV: We can vectorize this loop!
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %red.05 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds [255 x i32], ptr @a, i64 0, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %red.05
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 255
  br i1 %exitcond, label %for.end, label %for.body

; If it did, we have two loops:
; CHECK: vector.body:
; CHECK: br {{.*}} label %vector.body, !llvm.loop [[vect:![0-9]+]]
; CHECK: for.body:
; CHECK: br {{.*}} label %for.body{{.*}}, !llvm.loop [[scalar:![0-9]+]]

for.end:                                          ; preds = %for.body
  ret i32 %add
}

; Now, we check for the Hint metadata
; CHECK: [[vect]] = distinct !{[[vect]], [[width:![0-9]+]], [[runtime_unroll:![0-9]+]]}
; CHECK: [[width]] = !{!"llvm.loop.isvectorized", i32 1}
; CHECK: [[runtime_unroll]] = !{!"llvm.loop.unroll.runtime.disable"}
; CHECK: [[scalar]] = distinct !{[[scalar]], [[runtime_unroll]], [[width]]}
