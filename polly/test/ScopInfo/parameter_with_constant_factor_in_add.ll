; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
; Check that the access function of the store is simple and concise
;
; CHECK: p0: {0,+,(-1 + (sext i32 (-1 * %smax188) to i64))<nsw>}<%for.cond261.preheader>
;
; CHECK:      MustWriteAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:   [p_0] -> { Stmt_for_body276[i0] -> MemRef_A[p_0] };
;
; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @BPredPartitionCost(ptr %A) #0 {
entry:
  %curr_blk = alloca [16 x [16 x i32]], align 16
  br label %for.cond261.preheader.lr.ph

for.cond261.preheader.lr.ph:                      ; preds = %entry
  %smax188 = select i1 undef, i32 undef, i32 -9
  %0 = sub i32 0, %smax188
  %1 = sext i32 %0 to i64
  %2 = add i64 %1, -1
  br label %for.cond261.preheader

for.cond261.preheader:                            ; preds = %for.inc299, %for.cond261.preheader.lr.ph
  %indvars.iv189 = phi i64 [ 0, %for.cond261.preheader.lr.ph ], [ %indvars.iv.next190, %for.inc299 ]
  br i1 undef, label %for.cond273.preheader, label %for.inc299

for.cond273.preheader:                            ; preds = %for.cond261.preheader
  br label %for.body276

for.body276:                                      ; preds = %for.body276, %for.cond273.preheader
  %indvars.iv = phi i64 [ 0, %for.cond273.preheader ], [ %indvars.iv.next, %for.body276 ]
  %3 = add nsw i64 0, %indvars.iv189
  %arrayidx282 = getelementptr inbounds [16 x [16 x i32]], ptr %curr_blk, i64 0, i64 %3, i64 0
  %4 = load i32, ptr %arrayidx282, align 4
  %arridx = getelementptr i32, ptr %A, i64 %3
  store i32 0, ptr %arridx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br i1 false, label %for.body276, label %for.end291

for.end291:                                       ; preds = %for.body276
  ret void

for.inc299:                                       ; preds = %for.cond261.preheader
  %indvars.iv.next190 = add i64 %indvars.iv189, %2
  br i1 undef, label %for.cond261.preheader, label %if.end302

if.end302:                                        ; preds = %for.inc299
  ret void
}
