; RUN: opt < %s -basicaa -gvn -S | FileCheck %s

; CHECK-NOT: load
; CHECK-NOT: phi

define i8* @cat(i8* %s1, ...) nounwind {
entry:
  br i1 undef, label %bb, label %bb3

bb:                                               ; preds = %entry
  unreachable

bb3:                                              ; preds = %entry
  store i8* undef, i8** undef, align 4
  br i1 undef, label %bb5, label %bb6

bb5:                                              ; preds = %bb3
  unreachable

bb6:                                              ; preds = %bb3
  br label %bb12

bb8:                                              ; preds = %bb12
  br i1 undef, label %bb9, label %bb10

bb9:                                              ; preds = %bb8
  %0 = load i8*, i8** undef, align 4                   ; <i8*> [#uses=0]
  %1 = load i8*, i8** undef, align 4                   ; <i8*> [#uses=0]
  br label %bb11

bb10:                                             ; preds = %bb8
  br label %bb11

bb11:                                             ; preds = %bb10, %bb9
  br label %bb12

bb12:                                             ; preds = %bb11, %bb6
  br i1 undef, label %bb8, label %bb13

bb13:                                             ; preds = %bb12
  ret i8* undef
}
