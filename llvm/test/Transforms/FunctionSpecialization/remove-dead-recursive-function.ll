; RUN: opt -passes="ipsccp<func-spec>" -funcspec-min-function-size=3 -S < %s | FileCheck %s

define i64 @main(i64 %x, i1 %flag) {
entry:
  br i1 %flag, label %plus, label %minus

plus:
  %tmp0 = call i64 @compute(i64 %x, ptr @plus)
  br label %merge

minus:
  %tmp1 = call i64 @compute(i64 %x, ptr @minus)
  br label %merge

merge:
  %tmp2 = phi i64 [ %tmp0, %plus ], [ %tmp1, %minus]
  ret i64 %tmp2
}

; CHECK-NOT: define internal i64 @compute(
;
; CHECK-LABEL: define internal i64 @compute.specialized.1(i64 %n, ptr %binop) {
; CHECK:  [[TMP0:%.+]] = call i64 @plus(i64 %n)
; CHECK:  [[TMP1:%.+]] = call i64 @compute.specialized.1(i64 [[TMP2:%.+]], ptr @plus)
; CHECK:  add nsw i64 [[TMP1]], [[TMP0]]
;
; CHECK-LABEL: define internal i64 @compute.specialized.2(i64 %n, ptr %binop) {
; CHECK:  [[TMP0:%.+]] = call i64 @minus(i64 %n)
; CHECK:  [[TMP1:%.+]] = call i64 @compute.specialized.2(i64 [[TMP2:%.+]], ptr @minus)
; CHECK:  add nsw i64 [[TMP1]], [[TMP0]]
;
define internal i64 @compute(i64 %n, ptr %binop) {
entry:
  %cmp = icmp sgt i64 %n, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %call = call i64 %binop(i64 %n)
  %sub = add nsw i64 %n, -1
  %call1 = call i64 @compute(i64 %sub, ptr %binop)
  %add2 = add nsw i64 %call1, %call
  br label %if.end

if.end:
  %result.0 = phi i64 [ %add2, %if.then ], [ 0, %entry ]
  ret i64 %result.0
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
