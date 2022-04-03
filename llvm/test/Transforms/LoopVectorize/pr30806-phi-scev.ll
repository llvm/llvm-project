; RUN: opt -S -indvars < %s | FileCheck %s

; Produced from the test-case:
;
; extern void foo(char *, unsigned , unsigned *);
; extern void bar(int *, long);
; extern char *processBuf(char *);
;
; extern unsigned theSize;
;
; void foo(char *buf, unsigned denominator, unsigned *flag) {
;   int incr = (int) (theSize / denominator);
;   int inx = 0;
;   while (*flag) {
;     int itmp = inx + incr;
;     int i = (int) theSize;
;     bar(&i, (long) itmp);
;     buf = processBuf(buf);
;     inx = itmp;
;   }
; }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@theSize = external local_unnamed_addr global i32, align 4

define void @foo(i8* %buf, i32 %denominator, i32* %flag) local_unnamed_addr {
entry:
  %i = alloca i32, align 4
  %0 = load i32, i32* @theSize, align 4
  %div = udiv i32 %0, %denominator
  %1 = load i32, i32* %flag, align 4
  %tobool5 = icmp eq i32 %1, 0
  br i1 %tobool5, label %while.end, label %while.body.lr.ph

while.body.lr.ph:                                 ; preds = %entry
  %2 = bitcast i32* %i to i8*
  br label %while.body

while.body:                                       ; preds = %while.body.lr.ph, %while.body
; Check that there are two PHIs followed by a 'sext' in the same block, and that
; the test does not crash.
; CHECK:        phi
; CHECK-NEXT:   phi
; CHECK-NEXT:   sext
  %buf.addr.07 = phi i8* [ %buf, %while.body.lr.ph ], [ %call, %while.body ]
  %inx.06 = phi i32 [ 0, %while.body.lr.ph ], [ %add, %while.body ]
  %add = add nsw i32 %inx.06, %div
  %3 = load i32, i32* @theSize, align 4
  store i32 %3, i32* %i, align 4
  %conv = sext i32 %add to i64
  call void @bar(i32* nonnull %i, i64 %conv)
  %call = call i8* @processBuf(i8* %buf.addr.07)
  %4 = load i32, i32* %flag, align 4
  %tobool = icmp eq i32 %4, 0
  br i1 %tobool, label %while.end.loopexit, label %while.body

while.end.loopexit:                               ; preds = %while.body
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  ret void
}

declare void @bar(i32*, i64) local_unnamed_addr
declare i8* @processBuf(i8*) local_unnamed_addr
