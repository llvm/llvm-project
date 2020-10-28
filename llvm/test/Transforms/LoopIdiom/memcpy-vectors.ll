; RUN: opt -loop-idiom -S <%s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

define void @memcpy_fixed_vec(i64* noalias %a, i64* noalias %b) local_unnamed_addr #1 {
; CHECK-LABEL: @memcpy_fixed_vec(
; CHECK: entry:
; CHECK: memcpy
; CHECK: vector.body
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds i64, i64* %a, i64 %index
  %1 = bitcast i64* %0 to <2 x i64>*
  %wide.load = load <2 x i64>, <2 x i64>* %1, align 8
  %2 = getelementptr inbounds i64, i64* %b, i64 %index
  %3 = bitcast i64* %2 to <2 x i64>*
  store <2 x i64> %wide.load, <2 x i64>* %3, align 8
  %index.next = add nuw nsw i64 %index, 2
  %4 = icmp eq i64 %index.next, 1024
  br i1 %4, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %vector.body
  ret void
}

define void @memcpy_scalable_vec(i64* noalias %a, i64* noalias %b) local_unnamed_addr #1 {
; CHECK-LABEL: @memcpy_scalable_vec(
; CHECK: entry:
; CHECK-NOT: memcpy
; CHECK: vector.body
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %0 = bitcast i64* %a to <vscale x 2 x i64>*
  %1 = getelementptr inbounds <vscale x 2 x i64>, <vscale x 2 x i64>* %0, i64 %index
  %wide.load = load <vscale x 2 x i64>, <vscale x 2 x i64>* %1, align 16
  %2 = bitcast i64* %b to <vscale x 2 x i64>*
  %3 = getelementptr inbounds <vscale x 2 x i64>, <vscale x 2 x i64>* %2, i64 %index
  store <vscale x 2 x i64> %wide.load, <vscale x 2 x i64>* %3, align 16
  %index.next = add nuw nsw i64 %index, 1
  %4 = icmp eq i64 %index.next, 1024
  br i1 %4, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %vector.body
  ret void
}
