; REQUIRES: asserts
; RUN: not --crash opt -passes=loop-vectorize -mtriple=riscv64 -mattr=+v -disable-output %s

define void @test(ptr %p, i64 %a, i8 %b) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %iv = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %add = add i32 %iv, 1
  %cmp.slt = icmp slt i32 %iv, 2
  %shl = shl i64 %a, 48
  %ashr = ashr i64 %shl, 52
  %trunc.i32 = trunc i64 %ashr to i32
  br i1 %cmp.slt, label %cond.false, label %for.body

cond.false:                                       ; preds = %for.cond
  %zext = zext i8 %b to i32
  br label %for.body

for.body:                                         ; preds = %cond.false, %for.cond
  %cond = phi i32 [ %trunc.i32, %for.cond ], [ %zext, %cond.false ]
  %shl.i32 = shl i32 %cond, 8
  %trunc = trunc i32 %shl.i32 to i8
  store i8 %trunc, ptr %p, align 1
  %cmp = icmp slt i32 %iv, 2
  br i1 %cmp, label %for.cond, label %exit

exit:                                             ; preds = %for.body
  ret void
}
