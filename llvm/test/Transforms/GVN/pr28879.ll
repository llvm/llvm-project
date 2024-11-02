; RUN: opt -passes=gvn <%s -S -o - | FileCheck %s

define void @f() {
entry:
  %a = alloca <7 x i1>, align 2
  store <7 x i1> undef, <7 x i1>* %a, align 2
; CHECK: store <7 x i1> undef, <7 x i1>*
  %0 = getelementptr inbounds <7 x i1>, <7 x i1>* %a, i64 0, i64 0
  %val = load i1, i1* %0, align 2
; CHECK: load i1, i1*
  br i1 %val, label %cond.true, label %cond.false

cond.true:
  ret void

cond.false:
  ret void
}

define <7 x i1> @g(<7 x i1>* %a) {
entry:
  %vec = load <7 x i1>, <7 x i1>* %a
; CHECK: load <7 x i1>, <7 x i1>*
  %0 = getelementptr inbounds <7 x i1>, <7 x i1>* %a, i64 0, i64 0
  %val = load i1, i1* %0, align 2
; CHECK: load i1, i1*
  br i1 %val, label %cond.true, label %cond.false

cond.true:
  ret <7 x i1> %vec

cond.false:
  ret <7 x i1> <i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>
}
