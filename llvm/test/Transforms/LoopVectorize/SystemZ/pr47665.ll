; REQUIRES: asserts
; RUN: not --crash opt -passes=loop-vectorize -mtriple=s390x -mcpu=z14 -disable-output %s

define void @test(ptr %p, i40 %a) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %for.body ]
  %shl = shl i40 %a, 24
  %ashr = ashr i40 %shl, 28
  %trunc = trunc i40 %ashr to i32
  %icmp.eq = icmp eq i32 %trunc, 0
  %zext = zext i1 %icmp.eq to i32
  %icmp.ult = icmp ult i32 0, %zext
  %or = or i1 %icmp.ult, true
  %icmp.sgt = icmp sgt i1 %or, false
  store i1 %icmp.sgt, ptr %p, align 1
  %iv.next = add i32 %iv, 1
  %cond = icmp ult i32 %iv.next, 10
  br i1 %cond, label %for.body, label %exit

exit:                                             ; preds = %for.body
  ret void
}
