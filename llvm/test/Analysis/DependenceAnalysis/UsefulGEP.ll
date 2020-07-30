; RUN: opt < %s -disable-output "-passes=print<da>" -aa-pipeline=basic-aa 2>&1
; RUN: opt < %s -analyze -basic-aa -da
;; Check this doesn't crash.
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;; struct s {
;;   int A[10][10];
;;   int C[10][10][10]; 
;; } S;

;; void dep_constraint_crash_test(int k,int N) {
;;   for( int i=0;i<N;i++)
;;     for( int j=0;j<N;j++)
;;       S.A[0][0] = S.C[0][0][k];
;; }


%struct.s = type { [10 x [10 x i32]], [10 x [10 x [10 x i32]]] }

@S = common global %struct.s zeroinitializer

define void @dep_constraint_crash_test(i32 %k, i32 %N) {
entry:
  %cmp12 = icmp sgt i32 %N, 0
  br i1 %cmp12, label %for.cond1.preheader.lr.ph, label %for.end6

for.cond1.preheader.lr.ph:                        
  %idxprom = sext i32 %k to i64
  %arrayidx = getelementptr inbounds %struct.s, %struct.s* @S, i64 0, i32 1, i64 0, i64 0, i64 %idxprom
  br label %for.body3.preheader

for.body3.preheader:                              
  %i.013 = phi i32 [ 0, %for.cond1.preheader.lr.ph ], [ %inc5, %for.inc4 ]
  br label %for.body3

for.body3:                                        
  %j.011 = phi i32 [ %inc, %for.body3 ], [ 0, %for.body3.preheader ]
  %0 = load i32, i32* %arrayidx
  store i32 %0, i32* getelementptr inbounds (%struct.s, %struct.s* @S, i64 0, i32 0, i64 0, i64 0)
  %inc = add nuw nsw i32 %j.011, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %for.inc4, label %for.body3

for.inc4:                                         
  %inc5 = add nuw nsw i32 %i.013, 1
  %exitcond14 = icmp eq i32 %inc5, %N
  br i1 %exitcond14, label %for.end6, label %for.body3.preheader

for.end6:                                         
  ret void
}
