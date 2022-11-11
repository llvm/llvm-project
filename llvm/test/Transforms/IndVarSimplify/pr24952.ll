; RUN: opt -passes=indvars -S < %s | FileCheck %s

declare void @use(i1)

define void @f() {
; CHECK-LABEL: @f(
 entry:
  %x = alloca i32
  %y = alloca i32
  br label %loop

 loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.inc, %loop ]
  %iv.inc = add i32 %iv, 1

  %x.gep = getelementptr i32, i32* %x, i32 %iv
  %eql = icmp eq i32* %x.gep, %y
; CHECK-NOT: @use(i1 true)
  call void @use(i1 %eql)

  ; %be.cond deliberately 'false' -- we want want the trip count to be 0.
  %be.cond = icmp ult i32 %iv, 0
  br i1 %be.cond, label %loop, label %leave

 leave:
  ret void
}
