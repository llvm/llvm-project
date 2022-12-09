; RUN: opt -passes=ipsccp -specialize-functions -func-specialization-size-threshold=3 -S < %s | FileCheck %s

; Checks for callsites that have been annotated with MinSize. We only expect
; specialisation for the call that does not have the attribute:
;
; CHECK: plus:
; CHECK:   %tmp0 = call i64 @compute.1(i64 %x, ptr @plus)
; CHECK:   br label %merge
; CHECK: minus:
; CHECK:   %tmp1 = call i64 @compute(i64 %x, ptr @minus) #0
; CHECK:   br label %merge
;
define i64 @main(i64 %x, i1 %flag) {
entry:
  br i1 %flag, label %plus, label %minus

plus:
  %tmp0 = call i64 @compute(i64 %x, ptr @plus)
  br label %merge

minus:
  %tmp1 = call i64 @compute(i64 %x, ptr @minus) #0
  br label %merge

merge:
  %tmp2 = phi i64 [ %tmp0, %plus ], [ %tmp1, %minus]
  ret i64 %tmp2
}

define internal i64 @compute(i64 %x, ptr %binop) {
entry:
  %tmp0 = call i64 %binop(i64 %x)
  ret i64 %tmp0
}

define internal i64 @plus(i64 %x) {
entry:
  %tmp0 = add i64 %x, 1
  ret i64 %tmp0
}

define internal i64 @minus(i64 %x) {
entry:
  %tmp0 = sub i64 %x, 1
  ret i64 %tmp0
}

attributes #0 = { minsize optsize }
