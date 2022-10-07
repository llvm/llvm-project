; RUN: opt < %s -loop-interchange -cache-line-size=64 -pass-remarks-missed='loop-interchange' -pass-remarks-output=%t -S \
; RUN:     -verify-dom-info -verify-loop-info -verify-loop-lcssa 2>&1
; RUN: FileCheck --input-file=%t --check-prefix=REMARKS %s

; REMARKS: --- !Passed
; REMARKS-NEXT: Pass:            loop-interchange
; REMARKS-NEXT: Name:            Interchanged
; REMARKS-NEXT: Function:        pr48212

define void @pr48212([5 x i32]* %filter) {
entry:
  br label %L1

L1:                                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %L1, %for.inc6
  %temp.04 = phi i32 [ undef, %L1 ], [ %temp.1.lcssa, %for.inc6 ]
  %k1.03 = phi i32 [ 0, %L1 ], [ %inc7, %for.inc6 ]
  br label %L2

L2:                                               ; preds = %for.body
  br label %for.body3

for.body3:                                        ; preds = %L2, %for.inc
  %temp.12 = phi i32 [ %temp.04, %L2 ], [ %add, %for.inc ]
  %k2.01 = phi i32 [ 0, %L2 ], [ %inc, %for.inc ]
  %idxprom = sext i32 %k2.01 to i64
  %arrayidx = getelementptr inbounds [5 x i32], [5 x i32]* %filter, i64 %idxprom
  %idxprom4 = sext i32 %k1.03 to i64
  %arrayidx5 = getelementptr inbounds [5 x i32], [5 x i32]* %arrayidx, i64 0, i64 %idxprom4
  %0 = load i32, i32* %arrayidx5
  %add = add nsw i32 %temp.12, %0
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %inc = add nsw i32 %k2.01, 1
  %cmp2 = icmp slt i32 %inc, 3
  br i1 %cmp2, label %for.body3, label %for.end

for.end:                                          ; preds = %for.inc
  %temp.1.lcssa = phi i32 [ %add, %for.inc ]
  br label %for.inc6

for.inc6:                                         ; preds = %for.end
  %inc7 = add nsw i32 %k1.03, 1
  %cmp = icmp slt i32 %inc7, 5
  br i1 %cmp, label %for.body, label %for.end8

for.end8:                                         ; preds = %for.inc6
  ret void
}


