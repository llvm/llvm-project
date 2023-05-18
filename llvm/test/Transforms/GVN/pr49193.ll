; RUN: opt -passes=gvn -S < %s | FileCheck %s

@a = external local_unnamed_addr global i32, align 4
@b = external local_unnamed_addr global i32, align 4

; Function Attrs: nounwind readnone
declare ptr @j() local_unnamed_addr #0

; CHECK: define {{.*}}@k()

define i64 @k() local_unnamed_addr {
bb:
  br i1 undef, label %bb10.preheader, label %bb3

bb10.preheader:                                   ; preds = %bb
  br label %bb13

bb3:                                              ; preds = %bb
  %i4 = load i32, ptr @a, align 4
  %i5.not = icmp eq i32 %i4, 0
  br label %bb7

bb7:                                              ; preds = %bb3
  %i8 = tail call ptr @j()
  br label %bb37

bb13:                                             ; preds = %bb34, %bb10.preheader
  br i1 undef, label %bb30thread-pre-split, label %bb16

bb16:                                             ; preds = %bb13
  %i17 = tail call ptr @j()
  br i1 undef, label %bb22thread-pre-split, label %bb37.loopexit

bb22thread-pre-split:                             ; preds = %bb16
  br label %bb27

bb27:                                             ; preds = %bb22thread-pre-split
  br i1 undef, label %bb30thread-pre-split, label %bb37.loopexit

bb30thread-pre-split:                             ; preds = %bb27, %bb13
  %i31.pr = load i32, ptr @a, align 4
  %i32.not2 = icmp eq i32 %i31.pr, 0
  br label %bb34

bb34:                                             ; preds = %bb30thread-pre-split
  br i1 undef, label %bb37.loopexit, label %bb13

bb37.loopexit:                                    ; preds = %bb34, %bb27, %bb16
  br label %bb37

bb37:                                             ; preds = %bb37.loopexit, %bb7
  %i38 = load i32, ptr @a, align 4
  store i32 %i38, ptr @b, align 4
  %i39 = tail call ptr @j()
  unreachable
}

attributes #0 = { nounwind readnone }
