; RUN: opt -S -O3 -force-vector-width=2 -force-vector-interleave=1 < %s | FileCheck --check-prefix=LOOPVEC %s
; RUN: opt -S -O3 -vectorize-loops=false -force-vector-width=2 -force-vector-interleave=1 < %s | FileCheck --check-prefix=NOLOOPVEC %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Make sure we can disable vectorization in opt.

; LOOPVEC:       add <2 x i32>
; NOLOOPVEC-NOT: add <2 x i32>

define i32 @vect(i32* %a) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %red.05 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, %red.05
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 255
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %add
}
