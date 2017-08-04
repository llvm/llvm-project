; RUN: opt %loadPolly -polly-scops -polly-invariant-load-hoisting=true -polly-ignore-aliasing -polly-process-unprofitable -analyze < %s | FileCheck %s
;
; CHECK: Region: %entry.split---%if.end
; CHECK:     Invariant Accesses: {
; CHECK:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK:                 [y, p_1_loaded_from_j] -> { Stmt_for_body[i0] -> MemRef_j[0] };
; CHECK:             Execution Context: [y, p_1_loaded_from_j] -> {  :  }
; CHECK:     }

; CHECK:            MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK:                [y, p_1_loaded_from_j] -> { Stmt_for_body5[i0] -> MemRef_p[p_1_loaded_from_j + i0] };
; CHECK:            MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK:                [y, p_1_loaded_from_j] -> { Stmt_for_body[i0] -> MemRef_p[p_1_loaded_from_j + i0] };


define void @a(i32 %y, i32* nocapture %p, i32* nocapture readonly %j) local_unnamed_addr #0 {
entry:
  br label %entry.split

entry.split:
  %tobool = icmp eq i32 %y, 0
  br i1 %tobool, label %for.body5, label %for.body

for.body:
  %i.024 = phi i32 [ %inc, %for.body ], [ 0, %entry.split ]
  %0 = load i32, i32* %j, align 4
  %add = add nsw i32 %0, %i.024
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i32, i32* %p, i64 %idxprom
  store i32 %i.024, i32* %arrayidx, align 4
  %inc = add nuw nsw i32 %i.024, 1
  %exitcond26 = icmp eq i32 %inc, 10000
  br i1 %exitcond26, label %if.end, label %for.body

for.body5:
  %i1.023 = phi i32 [ %inc10, %for.body5 ], [ 0, %entry.split ]
  %mul = shl nsw i32 %i1.023, 1
  %1 = load i32, i32* %j, align 4
  %add6 = add nsw i32 %1, %i1.023
  %idxprom7 = sext i32 %add6 to i64
  %arrayidx8 = getelementptr inbounds i32, i32* %p, i64 %idxprom7
  store i32 %mul, i32* %arrayidx8, align 4
  %inc10 = add nuw nsw i32 %i1.023, 1
  %exitcond = icmp eq i32 %inc10, 10000
  br i1 %exitcond, label %if.end, label %for.body5

if.end:
  ret void
}

