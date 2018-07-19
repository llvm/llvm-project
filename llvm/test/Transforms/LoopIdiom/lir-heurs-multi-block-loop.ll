;  RUN: opt -basicaa -loop-idiom -use-lir-code-size-heurs=true < %s -S | FileCheck %s

; When compiling for codesize we avoid idiom recognition for a
; multi-block loop unless it is one of
; - a loop_memset idiom, or
; - a memset/memcpy idiom in a nested loop.

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1)
@APPLES = common global i32 0, align 4
@ORANGES = common global i32 0, align 4

; LIR allowed: loop_memset idiom in multi-block loop.
; ===================================================
; CHECK-LABEL: @LoopMemset
; CHECK: for.body.preheader:
; CHECK: call void @llvm.memset
; CHECK: for.body:
;
define i32 @LoopMemset([2048 x i8]* noalias nocapture %DST, i32 %SIZE) local_unnamed_addr optsize {
entry:
  %cmp12 = icmp sgt i32 %SIZE, 0
  br i1 %cmp12, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.inc
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.inc ]
  %BASKET.013 = phi i32 [ %BASKET.1, %for.inc ], [ 0, %for.body.preheader ]
  %arraydecay = getelementptr inbounds [2048 x i8], [2048 x i8]* %DST, i64 %indvars.iv, i64 0
  tail call void @llvm.memset.p0i8.i64(i8* %arraydecay, i8 -1, i64 2048, i32 1, i1 false)
  %0 = trunc i64 %indvars.iv to i32
  %rem11 = and i32 %0, 1
  %cmp1 = icmp eq i32 %rem11, 0
  %1 = load i32, i32* @ORANGES, align 4
  %2 = load i32, i32* @APPLES, align 4
  br i1 %cmp1, label %if.then, label %if.else

if.else:                                          ; preds = %for.body
  %dec3 = add nsw i32 %2, -1
  store i32 %dec3, i32* @APPLES, align 4
  br label %for.inc

if.then:                                          ; preds = %for.body
  %dec = add nsw i32 %1, -1
  store i32 %dec, i32* @ORANGES, align 4
  br label %for.inc

for.inc:                                          ; preds = %if.then, %if.else
  %.pn = phi i32 [ %2, %if.then ], [ %1, %if.else ]
  %BASKET.1 = add nsw i32 %.pn, %BASKET.013
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, %SIZE
  br i1 %exitcond, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.inc
  %BASKET.1.lcssa = phi i32 [ %BASKET.1, %for.inc ]
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  %BASKET.0.lcssa = phi i32 [ 0, %entry ], [ %BASKET.1.lcssa, %for.end.loopexit ]
  ret i32 %BASKET.0.lcssa
}

; LIR allowed: memset idiom in multi-block nested loop,
; which is recognized as a loop_memset in its turn.
; =====================================================
; CHECK-LABEL: @NestedMemset_LoopMemset
; CHECK: for.cond1.preheader.preheader:
; CHECK: call void @llvm.memset
; CHECK: for.cond1.preheader:
;
define i32 @NestedMemset_LoopMemset([2046 x i8]* noalias nocapture %DST, i32 %SIZE) local_unnamed_addr optsize {
entry:
  %cmp25 = icmp sgt i32 %SIZE, 0
  br i1 %cmp25, label %for.cond1.preheader.preheader, label %for.end11

for.cond1.preheader.preheader:                    ; preds = %entry
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond1.preheader.preheader, %for.inc9
  %i.027 = phi i32 [ %inc10, %for.inc9 ], [ 0, %for.cond1.preheader.preheader ]
  %BASKET.026 = phi i32 [ %BASKET.2.lcssa, %for.inc9 ], [ 0, %for.cond1.preheader.preheader ]
  %idxprom4 = sext i32 %i.027 to i64
  %rem22 = and i32 %i.027, 1
  %cmp6 = icmp eq i32 %rem22, 0
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.inc
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.inc ]
  %BASKET.123 = phi i32 [ %BASKET.026, %for.cond1.preheader ], [ %BASKET.2, %for.inc ]
  %arrayidx5 = getelementptr inbounds [2046 x i8], [2046 x i8]* %DST, i64 %idxprom4, i64 %indvars.iv
  store i8 -1, i8* %arrayidx5, align 1
  %0 = load i32, i32* @APPLES, align 4
  %1 = load i32, i32* @ORANGES, align 4
  br i1 %cmp6, label %if.then, label %if.else

if.else:                                          ; preds = %for.body3
  %dec8 = add nsw i32 %0, -1
  store i32 %dec8, i32* @APPLES, align 4
  br label %for.inc

if.then:                                          ; preds = %for.body3
  %dec = add nsw i32 %1, -1
  store i32 %dec, i32* @ORANGES, align 4
  br label %for.inc

for.inc:                                          ; preds = %if.then, %if.else
  %.pn = phi i32 [ %0, %if.then ], [ %1, %if.else ]
  %BASKET.2 = add nsw i32 %.pn, %BASKET.123
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 2046
  br i1 %exitcond, label %for.body3, label %for.inc9

for.inc9:                                         ; preds = %for.inc
  %BASKET.2.lcssa = phi i32 [ %BASKET.2, %for.inc ]
  %inc10 = add nsw i32 %i.027, 1
  %cmp = icmp slt i32 %inc10, %SIZE
  br i1 %cmp, label %for.cond1.preheader, label %for.end11.loopexit

for.end11.loopexit:                               ; preds = %for.inc9
  %BASKET.2.lcssa.lcssa = phi i32 [ %BASKET.2.lcssa, %for.inc9 ]
  br label %for.end11

for.end11:                                        ; preds = %for.end11.loopexit, %entry
  %BASKET.0.lcssa = phi i32 [ 0, %entry ], [ %BASKET.2.lcssa.lcssa, %for.end11.loopexit ]
  ret i32 %BASKET.0.lcssa
}

; LIR avoided: memset idiom in multi-block top-level loop.
; ========================================================
; CHECK-LABEL: @Non_NestedMemset 
; CHECK-NOT: call void @llvm.memset
;
define i32 @Non_NestedMemset(i8* noalias nocapture %DST, i32 %SIZE) local_unnamed_addr optsize {
entry:
  %cmp12 = icmp sgt i32 %SIZE, 0
  br i1 %cmp12, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.inc
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.inc ]
  %BASKET.013 = phi i32 [ %BASKET.1, %for.inc ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i8, i8* %DST, i64 %indvars.iv
  store i8 -1, i8* %arrayidx, align 1
  %0 = trunc i64 %indvars.iv to i32
  %rem11 = and i32 %0, 1
  %cmp1 = icmp eq i32 %rem11, 0
  %1 = load i32, i32* @ORANGES, align 4
  %2 = load i32, i32* @APPLES, align 4
  br i1 %cmp1, label %if.then, label %if.else

if.else:                                          ; preds = %for.body
  %dec3 = add nsw i32 %2, -1
  store i32 %dec3, i32* @APPLES, align 4
  br label %for.inc

if.then:                                          ; preds = %for.body
  %dec = add nsw i32 %1, -1
  store i32 %dec, i32* @ORANGES, align 4
  br label %for.inc

for.inc:                                          ; preds = %if.then, %if.else
  %.pn = phi i32 [ %2, %if.then ], [ %1, %if.else ]
  %BASKET.1 = add nsw i32 %.pn, %BASKET.013
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, %SIZE
  br i1 %exitcond, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.inc
  %BASKET.1.lcssa = phi i32 [ %BASKET.1, %for.inc ]
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  %BASKET.0.lcssa = phi i32 [ 0, %entry ], [ %BASKET.1.lcssa, %for.end.loopexit ]
  ret i32 %BASKET.0.lcssa
}

