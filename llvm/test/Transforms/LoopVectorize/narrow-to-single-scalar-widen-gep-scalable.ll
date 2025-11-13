; RUN: not --crash opt -p loop-vectorize -force-vector-width=2 \
; RUN:  -force-target-supports-scalable-vectors=true \
; RUN:  -scalable-vectorization=preferred -S %s

define void @widengep_narrow(ptr %in, ptr noalias %p) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.in.off = getelementptr i8, ptr %in, i64 8
  %gep.in.iv = getelementptr i32, ptr %gep.in.off, i64 %iv
  store ptr %gep.in.iv, ptr %p
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv, 1024
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}
