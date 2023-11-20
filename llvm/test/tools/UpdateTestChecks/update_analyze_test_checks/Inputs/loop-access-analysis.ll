; RUN: opt -passes='print<access-info>' < %s -disable-output 2>&1 | FileCheck %s

define void @laa(ptr nocapture readonly %Base1, ptr nocapture readonly %Base2, ptr %Dest) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.Dest = getelementptr inbounds float, ptr %Dest, i64 %iv
  %l.Dest = load float, ptr %gep.Dest
  %cmp = fcmp une float %l.Dest, 0.0
  %gep.1 = getelementptr inbounds float, ptr %Base1, i64 %iv
  %gep.2 = getelementptr inbounds float, ptr %Base2, i64 %iv
  %select = select i1 %cmp, ptr %gep.1, ptr %gep.2
  %sink = load float, ptr %select, align 4
  store float %sink, ptr %gep.Dest, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 100
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

define void @test_brace_escapes(ptr noundef %arr) {
entry:
  br label %loop.1

loop.1:
  %iv = phi i64 [ %iv.next, %loop.1 ], [ 8, %entry ]
  %arr.addr.0.i = phi ptr [ %incdec.ptr.i, %loop.1 ], [ %arr, %entry ]
  %incdec.ptr.i = getelementptr inbounds ptr, ptr %arr.addr.0.i, i64 1
  %0 = load ptr, ptr %arr.addr.0.i, align 8
  %tobool.not.i = icmp eq ptr %0, null
  %iv.next = add i64 %iv, 8
  br i1 %tobool.not.i, label %loop.1.exit, label %loop.1

loop.1.exit:
  %iv.lcssa = phi i64 [ %iv, %loop.1 ]
  br label %loop.2

loop.2:
  %iv.1 = phi i64 [ 0, %loop.1.exit ], [ %iv.1.next, %loop.2 ]
  %iv.2 = phi i64 [ %iv.lcssa, %loop.1.exit ], [ %iv.2.next, %loop.2 ]
  %gep.iv.1 = getelementptr inbounds ptr, ptr %arr, i64 %iv.1
  %l.1 = load ptr, ptr %gep.iv.1, align 8
  %iv.2.next = add nsw i64 %iv.2, 1
  %gep.iv.2 = getelementptr inbounds ptr, ptr %arr, i64 %iv.2
  store ptr %l.1, ptr %gep.iv.2, align 8
  %iv.1.next = add nuw nsw i64 %iv.1, 1
  %cmp = icmp ult i64 %iv.1.next, 1000
  br i1 %cmp, label %loop.2, label %exit

exit:
  ret void
}

