; RUN: opt %loadPolly -polly-print-ast -disable-output < %s | FileCheck %s
; XFAIL: *

;int bar1();
;int bar2();
;int bar3();
;int k;
;#define N 100
;int A[N];
;
;int foo (int z) {
;  int i, j;
;
;  for (i = 0; i < N; i++) {
;    A[i] = i;
;
;      for (j = 0; j < N * 2; j++)
;        A[i] = j * A[i];
;  }
;
;  return A[z];
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

@A = common global [100 x i32] zeroinitializer, align 4 ; <ptr> [#uses=2]
@k = common global i32 0, align 4                 ; <ptr> [#uses=0]

define i32 @foo(i32 %z) nounwind {
bb.nph31.split.us:
  br label %bb.nph.us

for.inc06.us:                                     ; preds = %for.body6.us
  store i32 %mul.us, ptr %arrayidx.us
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=2]
  %exitcond32 = icmp eq i64 %indvar.next, 100     ; <i1> [#uses=1]
  br i1 %exitcond32, label %for.end19, label %bb.nph.us

for.body6.us:                                     ; preds = %for.body6.us, %bb.nph.us
  %arrayidx10.tmp.0.us = phi i32 [ %i.027.us, %bb.nph.us ], [ %mul.us, %for.body6.us ] ; <i32> [#uses=1]
  %0 = phi i32 [ 0, %bb.nph.us ], [ %inc.us, %for.body6.us ] ; <i32> [#uses=2]
  %mul.us = mul i32 %arrayidx10.tmp.0.us, %0      ; <i32> [#uses=2]
  %inc.us = add nsw i32 %0, 1                     ; <i32> [#uses=2]
  %exitcond = icmp eq i32 %inc.us, 200            ; <i1> [#uses=1]
  br i1 %exitcond, label %for.inc06.us, label %for.body6.us

bb.nph.us:                                        ; preds = %bb.nph31.split.us, %for.inc06.us
  %indvar = phi i64 [ %indvar.next, %for.inc06.us ], [ 0, %bb.nph31.split.us ] ; <i64> [#uses=3]
  %arrayidx.us = getelementptr [100 x i32], ptr @A, i64 0, i64 %indvar ; <ptr> [#uses=2]
  %i.027.us = trunc i64 %indvar to i32            ; <i32> [#uses=2]
  store i32 %i.027.us, ptr %arrayidx.us
  br label %for.body6.us

for.end19:                                        ; preds = %for.inc06.us
  %idxprom21 = sext i32 %z to i64                 ; <i64> [#uses=1]
  %arrayidx22 = getelementptr inbounds [100 x i32], ptr @A, i64 0, i64 %idxprom21 ; <ptr> [#uses=1]
  %tmp23 = load i32, ptr %arrayidx22                  ; <i32> [#uses=1]
  ret i32 %tmp23
}
; CHECK: for (c2=0;c2<=99;c2++) {
; CHECK:   S{{[0-4]}}(c2);
; CHECK:   for (c4=0;c4<=199;c4++) {
; CHECK:     S{{[[0-4]}}(c2,c4);
; CHECK:   }
; CHECK:   S{{[0-4]}}(c2);
; CHECK: }

