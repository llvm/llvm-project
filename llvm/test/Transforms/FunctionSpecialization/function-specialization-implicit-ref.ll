; RUN: opt -passes="ipsccp<func-spec>" -S -force-specialization < %s 2>&1 | FileCheck %s

@a = global i32 1

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

define internal i64 @compute(i64 %x, ptr %binop) !implicit.ref !0 {
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

!0 = !{ptr @a}

; CHECK: @compute.specialized.1(i64 %x, ptr %binop) !implicit.ref !0
; CHECK: @compute.specialized.2(i64 %x, ptr %binop) !implicit.ref !0
