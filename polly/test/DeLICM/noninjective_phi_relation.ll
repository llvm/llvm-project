; RUN: opt %loadPolly -polly-delicm -analyze < %s | FileCheck %s -match-full-lines
;
; llvm.org/PR34485
; This produces a non-injective PHIRead -> PHIWrite map due to an invalid
; paramter assumption.  It does not map anything because the result does pass
; the non-conflicting test. It would be a better test if it did to test whether
; correct code is generated.
;
; unsigned a;
; long b, d;
; short c;
; int e, f;
; void fn1() {
;   for (;;) {
;     for (; f;)
;       for (;;)
;         ;
;     a -= 0 < b;
;     for (; f <= 5; f++) {
;       short *g = &c;
;       *g = a++ ? e *= 4 : c;
;       g = &f;
;       d = *g;
;     }
;   }
; }
;
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"

@f = common local_unnamed_addr global i32 0, align 4
@b = common local_unnamed_addr global i32 0, align 4
@a = common local_unnamed_addr global i32 0, align 4
@c = common local_unnamed_addr global i16 0, align 2
@e = common local_unnamed_addr global i32 0, align 4
@d = common local_unnamed_addr global i32 0, align 4

define void @fn1() local_unnamed_addr {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %.pr = load i32, i32* @f, align 4
  %tobool19 = icmp eq i32 %.pr, 0
  br i1 %tobool19, label %for.end.lr.ph, label %for.body

for.end.lr.ph:                                    ; preds = %entry.split
  %0 = load i32, i32* @b, align 4
  %cmp = icmp sgt i32 %0, 0
  %conv = zext i1 %cmp to i32
  %a.promoted20 = load i32, i32* @a, align 4
  br label %for.end

for.cond.for.body_crit_edge:                      ; preds = %for.end, %for.end12
  %inc.lcssa2125 = phi i32 [ %inc, %for.end12 ], [ %sub, %for.end ]
  store i32 %inc.lcssa2125, i32* @a, align 4
  br label %for.body

for.body:                                         ; preds = %for.cond.for.body_crit_edge, %entry.split
  br label %for.cond2

for.cond2:                                        ; preds = %for.cond2, %for.body
  br label %for.cond2

for.end:                                          ; preds = %for.end.lr.ph, %for.end12
  %inc.lcssa22 = phi i32 [ %a.promoted20, %for.end.lr.ph ], [ %inc, %for.end12 ]
  %sub = sub i32 %inc.lcssa22, %conv
  %.pr15 = load i32, i32* @f, align 4
  %cmp416 = icmp slt i32 %.pr15, 6
  br i1 %cmp416, label %for.body6.lr.ph, label %for.cond.for.body_crit_edge

for.body6.lr.ph:                                  ; preds = %for.end
  %c.promoted = load i16, i16* @c, align 2
  br label %for.body6

for.body6:                                        ; preds = %for.body6.lr.ph, %cond.end
  %conv918 = phi i16 [ %c.promoted, %for.body6.lr.ph ], [ %conv9, %cond.end ]
  %inc17 = phi i32 [ %sub, %for.body6.lr.ph ], [ %inc, %cond.end ]
  %1 = phi i32 [ %.pr15, %for.body6.lr.ph ], [ %inc11, %cond.end ]
  %inc = add i32 %inc17, 1
  %tobool7 = icmp eq i32 %inc17, 0
  br i1 %tobool7, label %cond.false, label %cond.true

cond.true:                                        ; preds = %for.body6
  %2 = load i32, i32* @e, align 4
  %mul = shl nsw i32 %2, 2
  store i32 %mul, i32* @e, align 4
  br label %cond.end

cond.false:                                       ; preds = %for.body6
  %conv8 = sext i16 %conv918 to i32
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %mul, %cond.true ], [ %conv8, %cond.false ]
  %conv9 = trunc i32 %cond to i16
  %3 = load i16, i16* bitcast (i32* @f to i16*), align 4
  %inc11 = add nsw i32 %1, 1
  store i32 %inc11, i32* @f, align 4
  %cmp4 = icmp slt i32 %1, 5
  br i1 %cmp4, label %for.body6, label %for.end12

for.end12:                                        ; preds = %cond.end
  %conv10 = sext i16 %3 to i32
  store i16 %conv9, i16* @c, align 2
  store i32 %conv10, i32* @d, align 4
  %tobool = icmp eq i32 %inc11, 0
  br i1 %tobool, label %for.end, label %for.cond.for.body_crit_edge
}


; CHECK: Statistics {
; CHECK:     Compatible overwrites: 1
; CHECK: }

; CHECK:  No modification has been made
