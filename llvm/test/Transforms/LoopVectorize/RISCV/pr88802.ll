; REQUIRES: asserts
; RUN: not --crash opt -passes=loop-vectorize -mtriple=riscv64 -mattr=+v -disable-output %s

define void @test(ptr %p) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %iv = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %add = add i32 %iv, 1
  %cmp.slt = icmp slt i32 %iv, 2
  br i1 %cmp.slt, label %cond.false, label %cond.true

cond.true:                                        ; preds = %for.cond
  %trunc.i32 = trunc i64 0 to i32
  br label %for.body

cond.false:                                       ; preds = %for.cond
  %zext = zext i8 0 to i32
  br label %for.body

for.body:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %trunc.i32, %cond.true ], [ %zext, %cond.false ]
  %cond.i8 = trunc i32 %cond to i8
  %and = and i8 %cond.i8, 0
  store i8 %and, ptr %p, align 1
  %cmp = icmp slt i32 %iv, 2
  br i1 %cmp, label %for.cond, label %exit

exit:                                             ; preds = %for.body
  ret void
}
