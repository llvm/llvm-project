; RUN: opt %loadPolly -polly-codegen -polly-ast-detect-parallel -S < %s | FileCheck %s
;
; Check that we mark multiple parallel loops correctly including the memory instructions.
;
; CHECK-DAG:  %polly.loop_cond[[COuter:[0-9]*]] = icmp sle i64 %polly.indvar_next{{[0-9]*}}, 1023
; CHECK-DAG:  br i1 %polly.loop_cond[[COuter]], label %polly.loop_header{{[0-9]*}}, label %polly.loop_exit{{[0-9]*}}, !llvm.loop ![[IDOuter:[0-9]*]]
;
; CHECK-DAG:  %polly.loop_cond[[CInner:[0-9]*]] = icmp sle i64 %polly.indvar_next{{[0-9]*}}, 511
; CHECK-DAG:  br i1 %polly.loop_cond[[CInner]], label %polly.loop_header{{[0-9]*}}, label %polly.loop_exit{{[0-9]*}}, !llvm.loop ![[IDInner:[0-9]*]]
;
; CHECK-DAG: store i32 %{{[a-z_0-9]*}}, i32* %{{[a-z_0-9]*}}, {{[ ._!,a-zA-Z0-9]*}}, !llvm.mem.parallel_loop_access !4
;
; CHECK-DAG: ![[IDOuter]] = distinct !{![[IDOuter]]}
; CHECK-DAG: ![[IDInner]] = distinct !{![[IDInner]]}
; CHECK-DAG: !4 = !{![[IDOuter]], ![[IDInner]]}
;
;    void jd(int *A) {
;      for (int i = 0; i < 1024; i++)
;        for (int j = 0; j < 512; j++)
;          A[i * 512 + j] = i + j;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(i32* %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc5, %entry
  %indvars.iv3 = phi i64 [ %indvars.iv.next4, %for.inc5 ], [ 0, %entry ]
  %exitcond6 = icmp ne i64 %indvars.iv3, 1024
  br i1 %exitcond6, label %for.body, label %for.end7

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %for.body ]
  %exitcond = icmp ne i64 %indvars.iv, 512
  br i1 %exitcond, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %tmp = add nsw i64 %indvars.iv3, %indvars.iv
  %tmp7 = shl nsw i64 %indvars.iv3, 9
  %tmp8 = add nsw i64 %tmp7, %indvars.iv
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %tmp8
  %tmp9 = trunc i64 %tmp to i32
  store i32 %tmp9, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.inc5

for.inc5:                                         ; preds = %for.end
  %indvars.iv.next4 = add nuw nsw i64 %indvars.iv3, 1
  br label %for.cond

for.end7:                                         ; preds = %for.cond
  ret void
}
