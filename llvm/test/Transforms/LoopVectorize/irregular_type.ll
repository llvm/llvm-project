; RUN: opt %s -passes=loop-vectorize -force-vector-width=4 -S | FileCheck %s

; Ensure the array loads/stores are not optimized into vector operations when
; the element type has padding bits.

; CHECK: foo
; CHECK: vector.body
; CHECK-NOT: load <4 x i7>
; CHECK-NOT: store <4 x i7>
; CHECK: for.body
define void @foo(ptr %a, i64 %n) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i7, ptr %a, i64 %indvars.iv
  %0 = load i7, ptr %arrayidx, align 1
  %sub = add nuw nsw i7 %0, 0
  store i7 %sub, ptr %arrayidx, align 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp = icmp eq i64 %indvars.iv.next, %n
  br i1 %cmp, label %for.exit, label %for.body

for.exit:
  ret void
}
