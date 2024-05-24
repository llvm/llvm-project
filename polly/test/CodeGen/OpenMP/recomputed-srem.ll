; RUN: opt %loadNPMPolly -polly-stmt-granularity=bb -passes=polly-codegen -polly-parallel \
; RUN: -polly-parallel-force -S < %s | FileCheck %s
;
; Test to verify that we pass %rem96 to the parallel subfunction.
;
; CHECK:       %[[R:[0-9]*]] = getelementptr inbounds { i32, i32, i64, ptr, ptr, i32 }, ptr %polly.par.userContext, i32 0, i32 5
; CHECK-NEXT:  %polly.subfunc.arg.rem96 = load i32, ptr %[[R]]
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @dmxpy(i32 %n1, ptr %y, i32 %n2, ptr %x) #0 {
entry:
  %rem96 = srem i32 %n2, 16
  %0 = sext i32 %rem96 to i64
  %1 = add i64 %0, 15
  br label %for.cond195.preheader

for.cond195.preheader:                            ; preds = %for.inc363, %entry
  %indvars.iv262 = phi i64 [ %1, %entry ], [ %indvars.iv.next263, %for.inc363 ]
  %j.0236 = phi i32 [ 0, %entry ], [ %add364, %for.inc363 ]
  br label %for.body197

for.body197:                                      ; preds = %for.body197, %for.cond195.preheader
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body197 ], [ 0, %for.cond195.preheader ]
  %arrayidx199 = getelementptr inbounds float, ptr %y, i64 %indvars.iv
  %2 = add nsw i64 %indvars.iv262, -6
  %arrayidx292 = getelementptr inbounds float, ptr %x, i64 %2
  %3 = load float, ptr %arrayidx292, align 4
  store float undef, ptr %arrayidx199, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, %n1
  br i1 %exitcond, label %for.body197, label %for.inc363

for.inc363:                                       ; preds = %for.body197
  %add364 = add nsw i32 %j.0236, 16
  %cmp193 = icmp slt i32 %add364, %n2
  %indvars.iv.next263 = add i64 %indvars.iv262, 16
  br i1 %cmp193, label %for.cond195.preheader, label %for.end365

for.end365:                                       ; preds = %for.inc363
  ret void
}
