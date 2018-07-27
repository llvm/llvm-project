; RUN: opt -S -debugify -loop-vectorize -force-vector-width=2 -force-vector-interleave=1 < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Test that the new vectorized loop has proper debug location.

define i32 @vect(i32* %a) {
entry:
  br label %for.body

; CHECK-LABEL: vector.body:
; CHECK: [[index:%.*]] = phi i64 {{.*}}, !dbg ![[line2:[0-9]+]]

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %red.05 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, %red.05
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 255
  br i1 %exitcond, label %for.end, label %for.body

; CHECK-LABEL: middle.block:
; CHECK: %cmp.n = icmp {{.*}}, !dbg ![[line1:[0-9]+]]
; CHECK: br i1 %cmp.n, {{.*}}, !dbg ![[line1]]

for.end:
  ret i32 %add
}

; CHECK: ![[line1]] = !DILocation(line: 1
; CHECK: ![[line2]] = !DILocation(line: 2
