; RUN: opt %loadPolly -polly-print-instructions -polly-print-scops -disable-output < %s | FileCheck %s
;
; CHECK:    Statements {
; CHECK-NEXT: 	Stmt_Stmt
; CHECK-NEXT:       Domain :=
; CHECK-NEXT:           { Stmt_Stmt[i0] : 0 <= i0 <= 1023 };
; CHECK-NEXT:       Schedule :=
; CHECK-NEXT:           { Stmt_Stmt[i0] -> [i0] };
; CHECK-NEXT:       MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:           { Stmt_Stmt[i0] -> MemRef_A[i0] };
; CHECK-NEXT:       Instructions {
; CHECK-NEXT:             store i32 %i.0, i32* %arrayidx, align 4, !polly_split_after !0
; CHECK-NEXT:       }
; CHECK-NEXT:   }
;
; Function Attrs: noinline nounwind uwtable
define void @func(i32* %A) #0 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %add, %for.inc ]
  %cmp = icmp slt i32 %i.0, 1024
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  br label %Stmt

Stmt:
  %idxprom = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom
  store i32 %i.0, i32* %arrayidx, align 4, !polly_split_after !0
  br label %for.inc

for.inc:                                          ; preds = %Stmt
  %add = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

!0 = !{!"polly_split_after"}
