; RUN: opt -S -passes=indvars < %s | FileCheck %s

target datalayout = "e-p:64:64"

define ptr @callbr_result_exit_value(ptr %a0) {
; CHECK-LABEL: define ptr @callbr_result_exit_value(
; CHECK-SAME: ptr [[A0:%.*]]) {
; CHECK:         [[A02:%.*]] = ptrtoint ptr [[A0]] to i64
; CHECK:         [[OUT:%.*]] = callbr ptr asm "", "=r,!i"()
; CHECK:         [[OUT1:%.*]] = ptrtoint ptr [[OUT]] to i64
; CHECK-NEXT:    [[DIFF:%.*]] = sub i64 [[OUT1]], [[A02]]
; CHECK:         [[SCEVGEP:%.*]] = getelementptr i8, ptr [[A0]], i64 [[DIFF]]
; CHECK-NEXT:    ret ptr [[SCEVGEP]]
;
entry:
  %out = callbr ptr asm "", "=r,!i"()
  to label %d [label %indirect]

indirect:
  ret ptr null

d:
  %cmp0 = icmp eq ptr %a0, %out
  br i1 %cmp0, label %end, label %loop

loop:
  %iv = phi ptr [ %a0, %d ], [ %inc, %loop ]
  %inc = getelementptr inbounds i8, ptr %iv, i64 1
  %cmp = icmp eq ptr %inc, %out
  br i1 %cmp, label %exit, label %loop

exit:
  %lcssa = phi ptr [ %inc, %loop ]
  ret ptr %lcssa

end:
  ret ptr %a0
}
