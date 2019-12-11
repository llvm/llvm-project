; RUN: opt < %s -ipconstprop -S | FileCheck %s

;; Check that the 21 constants got propagated properly
; CHECK: %M = add i32 21, 21
;; Check that the second return values didn't get propagated
; CHECK: %N = add i32 %B, %D

%0 = type { i32, i32 }

define internal %0 @foo(i1 %Q) {
  br i1 %Q, label %T, label %F

T:                                                ; preds = %0
  %mrv = insertvalue %0 undef, i32 21, 0
  %mrv1 = insertvalue %0 %mrv, i32 22, 1
  ret %0 %mrv1

F:                                                ; preds = %0
  %mrv2 = insertvalue %0 undef, i32 21, 0
  %mrv3 = insertvalue %0 %mrv2, i32 23, 1
  ret %0 %mrv3
}

define internal %0 @bar(i1 %Q) {
  %A = insertvalue %0 undef, i32 21, 0
  br i1 %Q, label %T, label %F

T:                                                ; preds = %0
  %B = insertvalue %0 %A, i32 22, 1
  ret %0 %B

F:                                                ; preds = %0
  %C = insertvalue %0 %A, i32 23, 1
  ret %0 %C
}

define %0 @caller(i1 %Q) {
  %X = call %0 @foo(i1 %Q)
  %A = extractvalue %0 %X, 0
  %B = extractvalue %0 %X, 1
  %Y = call %0 @bar(i1 %Q)
  %C = extractvalue %0 %Y, 0
  %D = extractvalue %0 %Y, 1
  %M = add i32 %A, %C
  %N = add i32 %B, %D
  ret %0 %X
}
