; RUN: not --crash opt -p loop-vectorize -force-vector-width=4 -S %s

define void @induction_with_multiple_instructions_in_chain(ptr %p, ptr noalias %q) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
  %ind.1 = phi i32 [ %ind.1.next, %loop ], [ 3, %entry ]
  %ind.2 = phi i32 [ %ind.1, %loop ], [ 0, %entry ]
  %sext.1 = sext i32 %ind.1 to i64
  %gep.1 = getelementptr i8, ptr %p, i64 %sext.1
  store i8 0, ptr %gep.1
  %sext.2 = sext i32 %ind.2 to i64
  %gep.2 = getelementptr i8, ptr %q, i64 %sext.2
  store i8 0, ptr %gep.2
  %iv.next = add i64 %iv, 1
  %ind.1.next = add i32 %ind.1, 3
  %ec = icmp eq i64 %iv, 1024
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}
