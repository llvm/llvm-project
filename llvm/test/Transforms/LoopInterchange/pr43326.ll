; RUN: opt < %s -basic-aa -loop-interchange -cache-line-size=64 -pass-remarks-missed='loop-interchange' -pass-remarks-output=%t -S \
; RUN:     -verify-dom-info -verify-loop-info -verify-loop-lcssa -stats 2>&1
; RUN: FileCheck --input-file=%t --check-prefix=REMARKS %s

@a = global i32 0
@b = global i8 0
@c = global i32 0
@d = global i32 0
@e = global [1 x [1 x i32]] zeroinitializer

; REMARKS: --- !Passed
; REMARKS-NEXT: Pass:            loop-interchange
; REMARKS-NEXT: Name:            Interchanged
; REMARKS-NEXT: Function:        pr43326

define void @pr43326() {
entry:
  %0 = load i32, i32* @a
  %tobool.not2 = icmp eq i32 %0, 0
  br i1 %tobool.not2, label %for.end14, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %d.promoted = load i32, i32* @d
  %a.promoted = load i32, i32* @a
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.inc12
  %inc1312 = phi i32 [ %a.promoted, %for.body.lr.ph ], [ %inc13, %for.inc12 ]
  %xor.lcssa.lcssa11 = phi i32 [ %d.promoted, %for.body.lr.ph ], [ %xor.lcssa.lcssa, %for.inc12 ]
  br label %for.body3

for.body3:                                        ; preds = %for.body, %for.inc10
  %xor.lcssa9 = phi i32 [ %xor.lcssa.lcssa11, %for.body ], [ %xor.lcssa, %for.inc10 ]
  %dec7 = phi i8 [ 0, %for.body ], [ %dec, %for.inc10 ]
  %idxprom8 = sext i8 %dec7 to i64
  br label %for.body7

for.body7:                                        ; preds = %for.body3, %for.inc
  %xor5 = phi i32 [ %xor.lcssa9, %for.body3 ], [ %xor, %for.inc ]
  %inc4 = phi i32 [ 0, %for.body3 ], [ %inc, %for.inc ]
  %idxprom = sext i32 %inc4 to i64
  %arrayidx9 = getelementptr inbounds [1 x [1 x i32]], [1 x [1 x i32]]* @e, i64 0, i64 %idxprom, i64 %idxprom8
  %1 = load i32, i32* %arrayidx9
  %xor = xor i32 %xor5, %1
  br label %for.inc

for.inc:                                          ; preds = %for.body7
  %inc = add nsw i32 %inc4, 1
  %cmp5 = icmp slt i32 %inc, 1
  br i1 %cmp5, label %for.body7, label %for.end

for.end:                                          ; preds = %for.inc
  %xor.lcssa = phi i32 [ %xor, %for.inc ]
  %inc.lcssa = phi i32 [ %inc, %for.inc ]
  br label %for.inc10

for.inc10:                                        ; preds = %for.end
  %dec = add i8 %dec7, -1
  %cmp = icmp sgt i8 %dec, -1
  br i1 %cmp, label %for.body3, label %for.end11

for.end11:                                        ; preds = %for.inc10
  %xor.lcssa.lcssa = phi i32 [ %xor.lcssa, %for.inc10 ]
  %dec.lcssa = phi i8 [ %dec, %for.inc10 ]
  %inc.lcssa.lcssa = phi i32 [ %inc.lcssa, %for.inc10 ]
  br label %for.inc12

for.inc12:                                        ; preds = %for.end11
  %inc13 = add nsw i32 %inc1312, 1
  %tobool.not = icmp eq i32 %inc13, 0
  br i1 %tobool.not, label %for.cond.for.end14_crit_edge, label %for.body

for.cond.for.end14_crit_edge:                     ; preds = %for.inc12
  %inc13.lcssa = phi i32 [ %inc13, %for.inc12 ]
  %inc.lcssa.lcssa.lcssa = phi i32 [ %inc.lcssa.lcssa, %for.inc12 ]
  %xor.lcssa.lcssa.lcssa = phi i32 [ %xor.lcssa.lcssa, %for.inc12 ]
  %dec.lcssa.lcssa = phi i8 [ %dec.lcssa, %for.inc12 ]
  store i8 %dec.lcssa.lcssa, i8* @b
  store i32 %xor.lcssa.lcssa.lcssa, i32* @d
  store i32 %inc.lcssa.lcssa.lcssa, i32* @c
  store i32 %inc13.lcssa, i32* @a
  br label %for.end14

for.end14:                                        ; preds = %for.cond.for.end14_crit_edge, %entry
  ret void
}
