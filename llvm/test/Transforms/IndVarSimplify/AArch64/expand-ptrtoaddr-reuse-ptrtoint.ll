; RUN: opt -passes='indvars' -S %s | FileCheck %s

target triple = "arm64-apple-macosx"

define i64 @count_then_convert(ptr %end) {
; CHECK-LABEL: @count_then_convert(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[END1:%.*]] = ptrtoint ptr [[END:%.*]] to i64
; CHECK-NEXT:    [[ISEMPTY:%.*]] = icmp eq ptr [[END]], null
; CHECK-NEXT:    br i1 [[ISEMPTY]], label [[EXIT:%.*]], label [[BODY_PH:%.*]]
; CHECK:       body.ph:
; CHECK-NEXT:    br label [[BODY:%.*]]
; CHECK:       body:
; CHECK-NEXT:    br i1 true, label [[EXIT_LOOPEXIT:%.*]], label [[BODY]]
; CHECK:       exit.loopexit:
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[END1]], -8
; CHECK-NEXT:    [[TMP1:%.*]] = lshr i64 [[TMP0]], 3
; CHECK-NEXT:    [[TMP2:%.*]] = add nuw nsw i64 [[TMP1]], 1
; CHECK-NEXT:    br label [[EXIT]]
; CHECK:       exit:
; CHECK-NEXT:    [[CNT:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ], [ [[TMP2]], [[EXIT_LOOPEXIT]] ]
; CHECK-NEXT:    [[ENDI:%.*]] = ptrtoint ptr [[END]] to i64
; CHECK-NEXT:    [[R:%.*]] = or i64 [[CNT]], [[ENDI]]
; CHECK-NEXT:    ret i64 [[R]]
;
entry:
  %isempty = icmp eq ptr %end, null
  br i1 %isempty, label %exit, label %body.ph

body.ph:
  br label %body

body:
  %p = phi ptr [ null, %body.ph ], [ %pn, %body ]
  %n = phi i64 [ 0, %body.ph ], [ %nn, %body ]
  %nn = add i64 %n, 1
  %pn = getelementptr inbounds nuw i8, ptr %p, i64 8
  %done = icmp eq ptr %pn, %end
  br i1 %done, label %exit, label %body

exit:
  %cnt = phi i64 [ 0, %entry ], [ %nn, %body ]
  %endi = ptrtoint ptr %end to i64
  %r = or i64 %cnt, %endi
  ret i64 %r
}
