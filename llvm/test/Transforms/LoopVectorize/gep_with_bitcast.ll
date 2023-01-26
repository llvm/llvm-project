; RUN: opt -S -passes=loop-vectorize,instcombine -force-vector-width=4  < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

; Vectorization of loop with bitcast between GEP and load
; Simplified source code:
;void foo (ptr __restrict__  in, bool * __restrict__ res) {
;
;  for (int i = 0; i < 4096; ++i)
;    res[i] = ((unsigned long long)in[i] == 0);
;}

; CHECK-LABEL: @foo
; CHECK: vector.body
; CHECK:  %[[IV:.+]] = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK:  %[[v0:.+]] = getelementptr inbounds ptr, ptr %in, i64 %[[IV]]
; CHECK:  %wide.load = load <4 x i64>, ptr %[[v0]], align 8
; CHECK:  icmp eq <4 x i64> %wide.load, zeroinitializer
; CHECK:  br i1

define void @foo(ptr noalias nocapture readonly %in, ptr noalias nocapture readnone %out, ptr noalias nocapture %res) #0 {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds ptr, ptr %in, i64 %indvars.iv
  %tmp54 = load i64, ptr %arrayidx, align 8
  %cmp1 = icmp eq i64 %tmp54, 0
  %arrayidx3 = getelementptr inbounds i8, ptr %res, i64 %indvars.iv
  %frombool = zext i1 %cmp1 to i8
  store i8 %frombool, ptr %arrayidx3, align 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 4096
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
