; RUN: opt -loop-idiom -mtriple=x86_64 -mcpu=core-avx2 < %s -S | FileCheck --check-prefix=ALL %s
; RUN: opt -loop-idiom -mtriple=x86_64 -mcpu=corei7 < %s -S | FileCheck --check-prefix=ALL %s

; Recognize CTTZ builtin pattern.
; Here it will replace the loop -
; assume builtin is always profitable.
;
; int cttz_zero_check(int n)
; {
;   int i = 0;
;   while(n) {
;     n <<= 1;
;     i++;
;   }
;   return i;
; }
;
; ALL-LABEL: @cttz_zero_check
; ALL:       %0 = call i32 @llvm.cttz.i32(i32 %n, i1 true)
; ALL-NEXT:  %1 = sub i32 32, %0
;
; Function Attrs: norecurse nounwind readnone uwtable
define i32 @cttz_zero_check(i32 %n) {
entry:
  %tobool4 = icmp eq i32 %n, 0
  br i1 %tobool4, label %while.end, label %while.body.preheader

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  %i.06 = phi i32 [ %inc, %while.body ], [ 0, %while.body.preheader ]
  %n.addr.05 = phi i32 [ %shl, %while.body ], [ %n, %while.body.preheader ]
  %shl = shl i32 %n.addr.05, 1
  %inc = add nsw i32 %i.06, 1
  %tobool = icmp eq i32 %shl, 0
  br i1 %tobool, label %while.end.loopexit, label %while.body

while.end.loopexit:                               ; preds = %while.body
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  %i.0.lcssa = phi i32 [ 0, %entry ], [ %inc, %while.end.loopexit ]
  ret i32 %i.0.lcssa
}

; Recognize CTTZ builtin pattern.
; Here it will replace the loop -
; assume builtin is always profitable.
;
; int cttz(int n)
; {
;   int i = 0;
;   while(n <<= 1) {
;     i++;
;   }
;   return i;
; }
;
; ALL-LABEL: @cttz
; ALL:      %0 = shl i32 %n, 1
; ALL-NEXT: %1 = call i32 @llvm.cttz.i32(i32 %0, i1 false)
; ALL-NEXT: %2 = sub i32 32, %1
; ALL-NEXT: %3 = add i32 %2, 1
;
; Function Attrs: norecurse nounwind readnone uwtable
define i32 @cttz(i32 %n) {
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %entry
  %n.addr.0 = phi i32 [ %n, %entry ], [ %shl, %while.cond ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %while.cond ]
  %shl = shl i32 %n.addr.0, 1
  %tobool = icmp eq i32 %shl, 0
  %inc = add nsw i32 %i.0, 1
  br i1 %tobool, label %while.end, label %while.cond

while.end:                                        ; preds = %while.cond
  ret i32 %i.0
}

