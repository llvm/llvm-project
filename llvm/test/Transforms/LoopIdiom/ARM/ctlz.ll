; RUN: opt -loop-idiom -mtriple=armv7a < %s -S | FileCheck -check-prefix=LZCNT --check-prefix=ALL %s
; RUN: opt -loop-idiom -mtriple=armv4t < %s -S | FileCheck -check-prefix=NOLZCNT --check-prefix=ALL %s

; Recognize CTLZ builtin pattern.
; Here we'll just convert loop to countable,
; so do not insert builtin if CPU do not support CTLZ
;
; int ctlz_and_other(int n, char *a)
; {
;   int i = 0, n0 = n;
;   while(n >>= 1) {
;     a[i] = (n0 & (1 << i)) ? 1 : 0;
;     i++;
;   }
;   return i;
; }
;
; LZCNT:  entry
; LZCNT:  %0 = call i32 @llvm.ctlz.i32(i32 %shr8, i1 true)
; LZCNT-NEXT:  %1 = sub i32 32, %0
; LZCNT-NEXT:  %2 = zext i32 %1 to i64
; LZCNT:  %indvars.iv.next.lcssa = phi i64 [ %2, %while.body ]
; LZCNT:  %4 = trunc i64 %indvars.iv.next.lcssa to i32
; LZCNT:  %i.0.lcssa = phi i32 [ 0, %entry ], [ %4, %while.end.loopexit ]
; LZCNT:  ret i32 %i.0.lcssa

; NOLZCNT:  entry
; NOLZCNT-NOT:  @llvm.ctlz

; Function Attrs: norecurse nounwind uwtable
define i32 @ctlz_and_other(i32 %n, i8* nocapture %a) {
entry:
  %shr8 = ashr i32 %n, 1
  %tobool9 = icmp eq i32 %shr8, 0
  br i1 %tobool9, label %while.end, label %while.body.preheader

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %while.body ], [ 0, %while.body.preheader ]
  %shr11 = phi i32 [ %shr, %while.body ], [ %shr8, %while.body.preheader ]
  %0 = trunc i64 %indvars.iv to i32
  %shl = shl i32 1, %0
  %and = and i32 %shl, %n
  %tobool1 = icmp ne i32 %and, 0
  %conv = zext i1 %tobool1 to i8
  %arrayidx = getelementptr inbounds i8, i8* %a, i64 %indvars.iv
  store i8 %conv, i8* %arrayidx, align 1
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %shr = ashr i32 %shr11, 1
  %tobool = icmp eq i32 %shr, 0
  br i1 %tobool, label %while.end.loopexit, label %while.body

while.end.loopexit:                               ; preds = %while.body
  %1 = trunc i64 %indvars.iv.next to i32
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  %i.0.lcssa = phi i32 [ 0, %entry ], [ %1, %while.end.loopexit ]
  ret i32 %i.0.lcssa
}

; Recognize CTLZ builtin pattern.
; Here it will replace the loop -
; assume builtin is always profitable.
;
; int ctlz_zero_check(int n)
; {
;   int i = 0;
;   while(n) {
;     n >>= 1;
;     i++;
;   }
;   return i;
; }
;
; ALL:  entry
; ALL:  %0 = call i32 @llvm.ctlz.i32(i32 %n, i1 true)
; ALL-NEXT:  %1 = sub i32 32, %0
; ALL:  %inc.lcssa = phi i32 [ %1, %while.body ]
; ALL:  %i.0.lcssa = phi i32 [ 0, %entry ], [ %inc.lcssa, %while.end.loopexit ]
; ALL:  ret i32 %i.0.lcssa

; Function Attrs: norecurse nounwind readnone uwtable
define i32 @ctlz_zero_check(i32 %n) {
entry:
  %tobool4 = icmp eq i32 %n, 0
  br i1 %tobool4, label %while.end, label %while.body.preheader

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  %i.06 = phi i32 [ %inc, %while.body ], [ 0, %while.body.preheader ]
  %n.addr.05 = phi i32 [ %shr, %while.body ], [ %n, %while.body.preheader ]
  %shr = ashr i32 %n.addr.05, 1
  %inc = add nsw i32 %i.06, 1
  %tobool = icmp eq i32 %shr, 0
  br i1 %tobool, label %while.end.loopexit, label %while.body

while.end.loopexit:                               ; preds = %while.body
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  %i.0.lcssa = phi i32 [ 0, %entry ], [ %inc, %while.end.loopexit ]
  ret i32 %i.0.lcssa
}

; Recognize CTLZ builtin pattern.
; Here it will replace the loop -
; assume builtin is always profitable.
;
; int ctlz(int n)
; {
;   int i = 0;
;   while(n >>= 1) {
;     i++;
;   }
;   return i;
; }
;
; ALL:  entry
; ALL:  %0 = ashr i32 %n, 1
; ALL-NEXT:  %1 = call i32 @llvm.ctlz.i32(i32 %0, i1 false)
; ALL-NEXT:  %2 = sub i32 32, %1
; ALL-NEXT:  %3 = add i32 %2, 1
; ALL:  %i.0.lcssa = phi i32 [ %2, %while.cond ]
; ALL:  ret i32 %i.0.lcssa

; Function Attrs: norecurse nounwind readnone uwtable
define i32 @ctlz(i32 %n) {
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %entry
  %n.addr.0 = phi i32 [ %n, %entry ], [ %shr, %while.cond ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %while.cond ]
  %shr = ashr i32 %n.addr.0, 1
  %tobool = icmp eq i32 %shr, 0
  %inc = add nsw i32 %i.0, 1
  br i1 %tobool, label %while.end, label %while.cond

while.end:                                        ; preds = %while.cond
  ret i32 %i.0
}

; Recognize CTLZ builtin pattern.
; Here it will replace the loop -
; assume builtin is always profitable.
;
; int ctlz_add(int n, int i0)
; {
;   int i = i0;
;   while(n >>= 1) {
;     i++;
;   }
;   return i;
; }
;
; ALL:  entry
; ALL:  %0 = ashr i32 %n, 1
; ALL-NEXT:  %1 = call i32 @llvm.ctlz.i32(i32 %0, i1 false)
; ALL-NEXT:  %2 = sub i32 32, %1
; ALL-NEXT:  %3 = add i32 %2, 1
; ALL-NEXT:  %4 = add i32 %2, %i0
; ALL:  %i.0.lcssa = phi i32 [ %4, %while.cond ]
; ALL:  ret i32 %i.0.lcssa
;
; Function Attrs: norecurse nounwind readnone uwtable
define i32 @ctlz_add(i32 %n, i32 %i0) {
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %entry
  %n.addr.0 = phi i32 [ %n, %entry ], [ %shr, %while.cond ]
  %i.0 = phi i32 [ %i0, %entry ], [ %inc, %while.cond ]
  %shr = ashr i32 %n.addr.0, 1
  %tobool = icmp eq i32 %shr, 0
  %inc = add nsw i32 %i.0, 1
  br i1 %tobool, label %while.end, label %while.cond

while.end:                                        ; preds = %while.cond
  ret i32 %i.0
}
