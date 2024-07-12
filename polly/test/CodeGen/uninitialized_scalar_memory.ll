; RUN: opt %loadNPMPolly -S -passes=polly-codegen < %s | FileCheck %s
;
; Verify we initialize the scalar locations reserved for the incoming phi
; values.
;
; CHECK:      polly.start:
; CHECK-NEXT:   store float %ebig.0, ptr %ebig.0.s2a
; CHECK-NEXT:   br label %polly.stmt.if.end.entry
;
;    int g(void);
;    float M;
;    int max(float *restrict xbig, int eres, int bres, float *restrict indx) {
;      int i, iebig;
;      float ebig;
;      for (i = 0; i < 4 + eres; i++) {
;        if (g())
;          break;
;
;        if (xbig[i] > ebig) {
;          ebig = xbig[i];
;          iebig = (int)(indx[i] + bres);
;        }
;      }
;      return (iebig);
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@M = common global float 0.000000e+00, align 4

define i32 @max(ptr noalias %xbig, i32 %eres, i32 %bres, ptr noalias %indx) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %iebig.0 = phi i32 [ undef, %entry ], [ %iebig.1, %for.inc ]
  %ebig.0 = phi float [ undef, %entry ], [ %ebig.1, %for.inc ]
  %add = add nsw i32 %eres, 4
  %tmp = sext i32 %add to i64
  %cmp = icmp slt i64 %indvars.iv, %tmp
  br i1 %cmp, label %for.body, label %for.end.loopexit

for.body:                                         ; preds = %for.cond
  %call = call i32 @g() #2
  %tobool = icmp eq i32 %call, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %for.body
  %iebig.0.lcssa1 = phi i32 [ %iebig.0, %for.body ]
  br label %for.end

if.end:                                           ; preds = %for.body
  %arrayidx = getelementptr inbounds float, ptr %xbig, i64 %indvars.iv
  %tmp3 = load float, ptr %arrayidx, align 4
  %cmp1 = fcmp ogt float %tmp3, %ebig.0
  br i1 %cmp1, label %if.then.2, label %if.end.9

if.then.2:                                        ; preds = %if.end
  %arrayidx4 = getelementptr inbounds float, ptr %xbig, i64 %indvars.iv
  %tmp4 = load float, ptr %arrayidx4, align 4
  %arrayidx6 = getelementptr inbounds float, ptr %indx, i64 %indvars.iv
  %tmp5 = load float, ptr %arrayidx6, align 4
  %conv = sitofp i32 %bres to float
  %add7 = fadd float %tmp5, %conv
  %conv8 = fptosi float %add7 to i32
  br label %if.end.9

if.end.9:                                         ; preds = %if.then.2, %if.end
  %iebig.1 = phi i32 [ %conv8, %if.then.2 ], [ %iebig.0, %if.end ]
  %ebig.1 = phi float [ %tmp4, %if.then.2 ], [ %ebig.0, %if.end ]
  br label %for.inc

for.inc:                                          ; preds = %if.end.9
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end.loopexit:                                 ; preds = %for.cond
  %iebig.0.lcssa = phi i32 [ %iebig.0, %for.cond ]
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %if.then
  %iebig.02 = phi i32 [ %iebig.0.lcssa, %for.end.loopexit ], [ %iebig.0.lcssa1, %if.then ]
  ret i32 %iebig.02
}

declare i32 @g() #1
