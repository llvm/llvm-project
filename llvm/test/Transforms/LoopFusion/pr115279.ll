; REQUIRES: asserts
; RUN: opt -passes=loop-fusion -disable-output -stats < %s 2>&1 | FileCheck -check-prefix=STAT %s
; STAT: 1 loop-fusion - Loops fused

; C Code
;
;;  for (i = 0; i < n; i++)
;;      array2[i] = array1[i] * 2;
;;  for (i = 0; i < n; i++)
;;      if (array1[i] % 2 == 0)
;;          array2[i] = array1[i] * array1[i];
; Loop fusion should not crash due to incorrect cached SCEV values as now
; forgetLoop() is used to reset them after the fusion.

define dso_local noundef i32 @forget_induction_scev(ptr noalias %array1, ptr noalias %array2) {
entry:
  br label %for.body

for.body:
  %indvars.iv54 = phi i64 [ %indvars.iv.next55, %for.body ], [ 0, %entry ]
  %arrayidx6 = getelementptr inbounds i32, ptr %array1, i64 %indvars.iv54
  %0 = load i32, ptr %arrayidx6, align 4
  %mul = shl nsw i32 %0, 1
  %arrayidx8 = getelementptr inbounds i32, ptr %array2, i64 %indvars.iv54
  store i32 %mul, ptr %arrayidx8, align 4
  %indvars.iv.next55 = add nuw nsw i64 %indvars.iv54, 1
  %exitcond57.not = icmp eq i64 %indvars.iv.next55, 1024
  br i1 %exitcond57.not, label %for.body14, label %for.body

for.body14:
  %indvars.iv58 = phi i64 [ %indvars.iv.next59, %for.inc26 ], [ 0, %for.body ]
  %arrayidx16 = getelementptr inbounds i32, ptr %array1, i64 %indvars.iv58
  %1 = load i32, ptr %arrayidx16, align 4
  %2 = and i32 %1, 1
  %cmp18 = icmp eq i32 %2, 0
  br i1 %cmp18, label %if.then, label %for.inc26

if.then:
  %mul23 = mul nsw i32 %1, %1
  %arrayidx25 = getelementptr inbounds i32, ptr %array2, i64 %indvars.iv58
  store i32 %mul23, ptr %arrayidx25, align 4
  br label %for.inc26

for.inc26:
  %indvars.iv.next59 = add nuw nsw i64 %indvars.iv58, 1
  %exitcond61.not = icmp eq i64 %indvars.iv.next59, 1024
  br i1 %exitcond61.not, label %for.end28, label %for.body14

for.end28:
  %arrayidx30 = getelementptr inbounds i8, ptr %array2, i64 4092
  %3 = load i32, ptr %arrayidx30, align 4
  ret i32 %3
}
