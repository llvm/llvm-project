; RUN: llc < %s -mtriple=x86_64-apple-darwin11 -relocation-model=pic -disable-fp-elim -stats 2>&1 | not grep ext-opt

define fastcc i8* @S_scan_str(i8* %start, i32 %keep_quoted, i32 %keep_delims) nounwind ssp {
entry:
  switch i8 undef, label %bb6 [
    i8 9, label %bb5
    i8 32, label %bb5
    i8 10, label %bb5
    i8 13, label %bb5
    i8 12, label %bb5
  ]

bb5:                                              ; preds = %entry, %entry, %entry, %entry, %entry
  br label %bb6

bb6:                                              ; preds = %bb5, %entry
  br i1 undef, label %bb7, label %bb9

bb7:                                              ; preds = %bb6
  unreachable

bb9:                                              ; preds = %bb6
  %0 = load i8* undef, align 1                    ; <i8> [#uses=3]
  br i1 undef, label %bb12, label %bb10

bb10:                                             ; preds = %bb9
  br i1 undef, label %bb12, label %bb11

bb11:                                             ; preds = %bb10
  unreachable

bb12:                                             ; preds = %bb10, %bb9
  br i1 undef, label %bb13, label %bb14

bb13:                                             ; preds = %bb12
  store i8 %0, i8* undef, align 1
  %1 = zext i8 %0 to i32                          ; <i32> [#uses=1]
  br label %bb18

bb14:                                             ; preds = %bb12
  br label %bb18

bb18:                                             ; preds = %bb14, %bb13
  %termcode.0 = phi i32 [ %1, %bb13 ], [ undef, %bb14 ] ; <i32> [#uses=2]
  %2 = icmp eq i8 %0, 0                           ; <i1> [#uses=1]
  br i1 %2, label %bb21, label %bb19

bb19:                                             ; preds = %bb18
  br i1 undef, label %bb21, label %bb20

bb20:                                             ; preds = %bb19
  br label %bb21

bb21:                                             ; preds = %bb20, %bb19, %bb18
  %termcode.1 = phi i32 [ %termcode.0, %bb18 ], [ %termcode.0, %bb19 ], [ undef, %bb20 ] ; <i32> [#uses=0]
  unreachable
}
