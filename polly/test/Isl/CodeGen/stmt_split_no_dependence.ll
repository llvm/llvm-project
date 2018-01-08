; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s
;
; CHECK:   store i32 %8, i32* %scevgep, align 4, !alias.scope !1, !noalias !3
; CHECK:   store i32 %9, i32* %scevgep4, align 4, !alias.scope !4, !noalias !5
;
;      void func(int *A, int *B){
;        for (int i = 0; i < 1024; i+=1) {
;      Stmt:
;          A[i] = i;
;          B[i] = i;
;        }
;      }
;
; Function Attrs: noinline nounwind uwtable
define void @func(i32* %A, i32* %B) #0 {
entry:
  br label %for.cond

for.cond: 					 ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %add, %for.inc ]
  %cmp = icmp slt i32 %i.0, 1024
  br i1 %cmp, label %for.body, label %for.end

for.body: 					 ; preds = %for.cond
  br label %Stmt

Stmt: 						 ; preds = %for.body
  %idxprom = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom
  store i32 %i.0, i32* %arrayidx, align 4, !polly_split_after !0
  %idxprom1 = sext i32 %i.0 to i64
  %arrayidx2 = getelementptr inbounds i32, i32* %B, i64 %idxprom1
  store i32 %i.0, i32* %arrayidx2, align 4
  br label %for.inc

for.inc: 					 ; preds = %Stmt
  %add = add nsw i32 %i.0, 1
  br label %for.cond

for.end: 					 ; preds = %for.cond
  ret void
}

!0 = !{!"polly_split_after"} 
