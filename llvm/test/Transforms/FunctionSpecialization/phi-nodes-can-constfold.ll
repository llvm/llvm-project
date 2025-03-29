; RUN: opt -passes="ipsccp<func-spec>" -funcspec-min-function-size=10 -funcspec-for-literal-constant -S < %s | FileCheck %s

define i64 @bar(i1 %c1, i1 %c2, i1 %c3, i1 %c4, i64 %x1) {
; CHECK-LABEL: define i64 @bar(
; CHECK-SAME: i1 [[C1:%.*]], i1 [[C2:%.*]], i1 [[C3:%.*]], i1 [[C4:%.*]], i64 [[X1:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[F1:%.*]] = call i64 @foo.specialized.1(i64 3, i1 [[C1]], i1 [[C2]], i1 [[C3]], i1 [[C4]])
; CHECK-NEXT:    [[F2:%.*]] = call i64 @foo(i64 [[X1]], i1 [[C1]], i1 [[C2]], i1 [[C3]], i1 [[C4]])
; CHECK-NEXT:    [[ADD:%.*]] = add i64 [[F1]], [[F2]]
; CHECK-NEXT:    ret i64 [[ADD]]
;
entry:
  %f1 = call i64 @foo(i64 3, i1 %c1, i1 %c2, i1 %c3, i1 %c4)
  %f2 = call i64 @foo(i64 %x1, i1 %c1, i1 %c2, i1 %c3, i1 %c4)
  %add = add i64 %f1, %f2
  ret i64 %add
}

define internal i64 @foo(i64 %n, i1 %c1, i1 %c2, i1 %c3, i1 %c4) {
entry:
  br label %l0
  
l1:
  %phi1 = phi i64 [ %phi0, %l0 ], [ %phi2, %l2 ]
  %add = add i64 %phi1, 1
  %div = sdiv i64 %add, 2
  br i1 %c2, label %l2, label %exit

l2:
  %phi2 = phi i64 [ %phi0, %l0 ], [ %phi1, %l1 ]
  %sub = sub i64 %phi2, 1
  %mul = mul i64 %sub, 2
  br i1 %c4, label %l1, label %exit

l0:
  %phi0 = phi i64 [ %n, %entry ]
  br i1 %c1, label %l1, label %l2

exit:
  %res = phi i64 [ %div, %l1 ], [ %mul, %l2]
  ret i64 %res
}
