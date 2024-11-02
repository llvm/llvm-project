; RUN: opt -passes=loop-vectorize -pass-remarks=loop-vectorize -S < %s 2>&1 | FileCheck %s

; FIXME: Check for -pass-remarks-missed and -pass-remarks-analysis output when
; addAcyclicInnerLoop emits analysis.

; Check that opt does not crash on such input:
;
; a, b, c;
; fn1() {
;   while (b--) {
;     c = a;
;     switch (a & 3)
;     case 0:
;       do
;     case 3:
;     case 2:
;     case 1:
;         ;
;         while (--c)
;           ;
;   }
; }

@b = common global i32 0, align 4
@a = common global i32 0, align 4
@c = common global i32 0, align 4

; CHECK-NOT: vectorized loop
; CHECK-LABEL: fn1

define i32 @fn1() {
entry:
  %tmp2 = load i32, i32* @b, align 4
  %dec3 = add nsw i32 %tmp2, -1
  store i32 %dec3, i32* @b, align 4
  %tobool4 = icmp eq i32 %tmp2, 0
  br i1 %tobool4, label %while.end, label %while.body.lr.ph

while.body.lr.ph:                                 ; preds = %entry
  %tmp1 = load i32, i32* @a, align 4
  %and = and i32 %tmp1, 3
  %switch = icmp eq i32 %and, 0
  br label %while.body

while.cond:                                       ; preds = %do.cond
  %dec = add nsw i32 %dec7, -1
  %tobool = icmp eq i32 %dec7, 0
  br i1 %tobool, label %while.cond.while.end_crit_edge, label %while.body

while.body:                                       ; preds = %while.body.lr.ph, %while.cond
  %dec7 = phi i32 [ %dec3, %while.body.lr.ph ], [ %dec, %while.cond ]
  br i1 %switch, label %do.body, label %do.cond

do.body:                                          ; preds = %do.cond, %while.body
  %dec25 = phi i32 [ %dec2, %do.cond ], [ %tmp1, %while.body ]
  br label %do.cond

do.cond:                                          ; preds = %do.body, %while.body
  %dec26 = phi i32 [ %dec25, %do.body ], [ %tmp1, %while.body ]
  %dec2 = add nsw i32 %dec26, -1
  %tobool3 = icmp eq i32 %dec2, 0
  br i1 %tobool3, label %while.cond, label %do.body

while.cond.while.end_crit_edge:                   ; preds = %while.cond
  store i32 0, i32* @c, align 4
  store i32 -1, i32* @b, align 4
  br label %while.end

while.end:                                        ; preds = %while.cond.while.end_crit_edge, %entry
  ret i32 undef
}
