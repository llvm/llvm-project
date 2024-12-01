; RUN: opt -passes=loop-vectorize %s -force-vector-width=1 -force-vector-interleave=2 -S -o - | FileCheck %s

define void @foo(ptr addrspace(1) %in) {
entry:
  br label %loop

loop:
  %iter = phi i64 [ %next, %loop ], [ 0, %entry ]
  %ascast = addrspacecast ptr addrspace(1) %in to ptr
  %next = add i64 %iter, 1
  %arrayidx = getelementptr inbounds i64, ptr %ascast, i64 %next
  store i64 %next, ptr %arrayidx, align 4

; check that we find the two interleaved blocks with ascast, gep and store:
; CHECK: pred.store.if:
; CHECK: [[ID1:%.*]] = add i64 %{{.*}}, 1
; CHECK: [[AS1:%.*]] = addrspacecast ptr addrspace(1) %{{.*}} to ptr
; CHECK: [[GEP1:%.*]] = getelementptr inbounds i64, ptr [[AS1]], i64 [[ID1]]
; CHECK: store i64 [[ID1]], ptr [[GEP1]]

; CHECK: pred.store.if1:
; CHECK: [[ID2:%.*]] = add i64 %{{.*}}, 1
; CHECK: [[AS2:%.*]] = addrspacecast ptr addrspace(1) %in to ptr
; CHECK: [[GEP2:%.*]] = getelementptr inbounds i64, ptr [[AS2]], i64 [[ID2]]
; CHECK: store i64 [[ID2]], ptr %9, align 4

  %cmp = icmp eq i64 %next, 7
  br i1 %cmp, label %exit, label %loop

; check that we branch to the exit block
; CHECK: middle.block:
; CHECK: br i1 true, label %exit, label %scalar.ph

exit:
  ret void
; CHECK: exit:
; CHECK:   ret void
}

; CHECK: !{{[0-9]*}} = !{!"llvm.loop.isvectorized", i32 1}
