; RUN: llc < %s -mtriple=thumbv7-apple-darwin -relocation-model=pic -disable-fp-elim -mcpu=cortex-a8

define void @get_initial_mb16x16_cost() nounwind {
entry:
  br i1 undef, label %bb4, label %bb1

bb1:                                              ; preds = %entry
  br label %bb7

bb4:                                              ; preds = %entry
  br i1 undef, label %bb7.thread, label %bb5

bb5:                                              ; preds = %bb4
  br label %bb7

bb7.thread:                                       ; preds = %bb4
  br label %bb8

bb7:                                              ; preds = %bb5, %bb1
  br i1 undef, label %bb8, label %bb10

bb8:                                              ; preds = %bb7, %bb7.thread
  %0 = phi double [ 5.120000e+02, %bb7.thread ], [ undef, %bb7 ] ; <double> [#uses=1]
  %1 = fdiv double %0, undef                      ; <double> [#uses=0]
  unreachable

bb10:                                             ; preds = %bb7
  ret void
}
