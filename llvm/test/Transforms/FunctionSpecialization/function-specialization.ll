; RUN: opt -passes="ipsccp<func-spec>" -funcspec-min-function-size=3 -S < %s | FileCheck %s
; RUN: opt -passes="ipsccp<no-func-spec>" -funcspec-min-function-size=3 -S < %s | FileCheck %s --check-prefix=NOFSPEC

define i64 @main(i64 %x, i1 %flag) {
;
; CHECK-LABEL: @main(i64 %x, i1 %flag) {
; CHECK:      entry:
; CHECK-NEXT:   br i1 %flag, label %plus, label %minus
; CHECK:      plus:
; CHECK-NEXT:   [[TMP0:%.+]] = call i64 @compute.specialized.1(i64 %x, ptr @plus)
; CHECK-NEXT:   br label %merge
; CHECK:      minus:
; CHECK-NEXT:   [[TMP1:%.+]] = call i64 @compute.specialized.2(i64 %x, ptr @minus)
; CHECK-NEXT:   br label %merge
; CHECK:      merge:
; CHECK-NEXT:   [[TMP2:%.+]] = phi i64 [ [[TMP0]], %plus ], [ [[TMP1]], %minus ]
; CHECK-NEXT:   ret i64 [[TMP2]]
; CHECK-NEXT: }
;
; NOFSPEC-LABEL: @main(i64 %x, i1 %flag) {
; NOFSPEC-NOT: call i64 @compute.specialized.{{[0-9]+}}(
; NOFSPEC: call i64 @compute(
;
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
; CHECK-LABEL: define internal i64 @compute.specialized.1(i64 %x, ptr %binop) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.+]] = call i64 @plus(i64 %x)
; CHECK-NEXT:    ret i64 [[TMP0]]
; CHECK-NEXT:  }
;
; CHECK-LABEL: define internal i64 @compute.specialized.2(i64 %x, ptr %binop) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.+]] = call i64 @minus(i64 %x)
; CHECK-NEXT:    ret i64 [[TMP0]]
; CHECK-NEXT:  }
;
; NOFSPEC: define internal i64 @compute(
; NOFSPEC-NOT: define internal i64 @compute.specialized.{{[0-9]+}}(
;
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
