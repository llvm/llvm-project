; REQUIRES: asserts
; RUN: not --crash opt -p loop-vectorize -S %s

target triple = "riscv64-unknown-linux-gnu"

define void @ptr_induction(ptr %p, ptr noalias %q, ptr noalias %p.end) #0 {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %ptr.ind = phi ptr [ %p, %entry ], [ %ptr.ind.next, %loop ]
  %ptri64 = ptrtoint ptr %ptr.ind to i64
  store i64 %ptri64, ptr %q
  store i64 %iv, ptr %p
  %iv.next = add i64 %iv, 1
  %ptr.ind.next = getelementptr i8, ptr %ptr.ind, i64 1
  %ec = icmp eq ptr %ptr.ind, %p.end
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

attributes #0 = { "target-features"="+v" }
