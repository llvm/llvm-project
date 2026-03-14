; RUN: opt -passes=loop-idiom -S <%s | FileCheck %s

define void @memcpy_fixed_vec(ptr noalias %a, ptr noalias %b) local_unnamed_addr #1 {
; CHECK-LABEL: @memcpy_fixed_vec(
; CHECK: entry:
; CHECK: memcpy
; CHECK: vector.body
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds i64, ptr %a, i64 %index
  %wide.load = load <2 x i64>, ptr %0, align 8
  %1 = getelementptr inbounds i64, ptr %b, i64 %index
  store <2 x i64> %wide.load, ptr %1, align 8
  %index.next = add nuw nsw i64 %index, 2
  %2 = icmp eq i64 %index.next, 1024
  br i1 %2, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %vector.body
  ret void
}

define void @memcpy_scalable_vec(ptr noalias %a, ptr noalias %b) local_unnamed_addr #1 {
; CHECK-LABEL: @memcpy_scalable_vec(
; CHECK: entry:
; CHECK-NOT: memcpy
; CHECK: vector.body
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds <vscale x 2 x i64>, ptr %a, i64 %index
  %wide.load = load <vscale x 2 x i64>, ptr %0, align 16
  %1 = getelementptr inbounds <vscale x 2 x i64>, ptr %b, i64 %index
  store <vscale x 2 x i64> %wide.load, ptr %1, align 16
  %index.next = add nuw nsw i64 %index, 1
  %2 = icmp eq i64 %index.next, 1024
  br i1 %2, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %vector.body
  ret void
}
