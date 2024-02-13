; RUN: opt -passes="ipsccp<func-spec>" -funcspec-min-function-size=10 -funcspec-for-literal-constant -S < %s | FileCheck %s

define i64 @bar(i1 %c1, i1 %c2, i1 %c3, i1 %c4, i64 %x1) {
; CHECK-LABEL: define i64 @bar(
; CHECK-SAME: i1 [[C1:%.*]], i1 [[C2:%.*]], i1 [[C3:%.*]], i1 [[C4:%.*]], i64 [[X1:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[F1:%.*]] = call i64 @foo(i64 3, i64 4, i1 [[C1]], i1 [[C2]], i1 [[C3]], i1 [[C4]])
; CHECK-NEXT:    [[F2:%.*]] = call i64 @foo(i64 4, i64 [[X1]], i1 [[C1]], i1 [[C2]], i1 [[C3]], i1 [[C4]])
; CHECK-NEXT:    [[F3:%.*]] = call i64 @foo.specialized.1(i64 3, i64 3, i1 [[C1]], i1 [[C2]], i1 [[C3]], i1 [[C4]])
; CHECK-NEXT:    [[ADD:%.*]] = add i64 [[F1]], [[F2]]
; CHECK-NEXT:    [[ADD2:%.*]] = add i64 [[ADD]], [[F3]]
; CHECK-NEXT:    ret i64 [[ADD2]]
;
entry:
  %f1 = call i64 @foo(i64 3, i64 4, i1 %c1, i1 %c2, i1 %c3, i1 %c4)
  %f2 = call i64 @foo(i64 4, i64 %x1, i1 %c1, i1 %c2, i1 %c3, i1 %c4)
  %f3 = call i64 @foo(i64 3, i64 3, i1 %c1, i1 %c2, i1 %c3, i1 %c4)
  %add = add i64 %f1, %f2
  %add2 = add i64 %add, %f3
  ret i64 %add2
}

define internal i64 @foo(i64 %n, i64 %m, i1 %c1, i1 %c2, i1 %c3, i1 %c4) {
entry:
  br i1 %c1, label %l1, label %l4

l1:
  %phi1 = phi i64 [ %n, %entry ], [ %phi2, %l2 ]
  %add = add i64 %phi1, 1
  %div = sdiv i64 %add, 2
  br i1 %c2, label %l1_5, label %exit

l1_5:
  br i1 %c3, label %l2, label %l3

l2:
  %phi2 = phi i64 [ %phi1, %l1_5 ], [ %phi3, %l3 ]
  br label %l1

l3:
  %phi3 = phi i64 [ %phi1, %l1_5 ], [ %m, %l4 ]
  br i1 %c2, label %l4, label %l2

l4:
  %phi4 = phi i64 [ %n, %entry ], [ %phi3, %l3 ]
  %sub = sub i64 %phi4, 1
  %mul = mul i64 %sub, 2
  br i1 %c4, label %l3, label %exit

exit:
  %res = phi i64 [ %div, %l1 ], [ %mul, %l4]
  ret i64 %res
}

