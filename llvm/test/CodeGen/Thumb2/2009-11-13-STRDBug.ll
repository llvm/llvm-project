; RUN: llc < %s -mtriple=thumbv7-apple-darwin10
; rdar://7394794

define void @lshift_double(i64 %l1, i64 %h1, i64 %count, i32 %prec, i64* nocapture %lv, i64* nocapture %hv, i32 %arith) nounwind {
entry:
  %..i = select i1 false, i64 0, i64 0            ; <i64> [#uses=1]
  br i1 undef, label %bb11.i, label %bb6.i

bb6.i:                                            ; preds = %entry
  %0 = lshr i64 %h1, 0                            ; <i64> [#uses=1]
  store i64 %0, i64* %hv, align 4
  %1 = lshr i64 %l1, 0                            ; <i64> [#uses=1]
  %2 = or i64 0, %1                               ; <i64> [#uses=1]
  store i64 %2, i64* %lv, align 4
  br label %bb11.i

bb11.i:                                           ; preds = %bb6.i, %entry
  store i64 %..i, i64* %lv, align 4
  ret void
}
