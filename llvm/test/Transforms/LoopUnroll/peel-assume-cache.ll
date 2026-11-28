; RUN: opt -passes='function(require<assumptions>,loop(loop-unroll-full),print<assumptions>)' -disable-output %s 2>&1 | FileCheck %s

; %prev is %init on the first iteration and %k afterwards, so loop-unroll-full peels one
; iteration. The peeled copy of the assume must be in the cache.

; CHECK-LABEL: Cached assumptions for function: f
; CHECK-DAG: %c = icmp ult i32 %k, 100
; CHECK-DAG: %c.peel = icmp ult i32 %init, 100

define void @f(ptr %a, i64 %n, i32 %init, i32 %k) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %prev = phi i32 [ %init, %entry ], [ %k, %loop ]
  %c = icmp ult i32 %prev, 100
  call void @llvm.assume(i1 %c)
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  store i32 %prev, ptr %p, align 4
  %i.next = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.next, %n
  br i1 %more, label %loop, label %exit

exit:
  ret void
}

declare void @llvm.assume(i1)
