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
