; RUN: opt -passes="loop-idiom" < %s -S | FileCheck %s

; Enzyme type annotations on loads/stores are kept as "enzyme_truetype" on
; the formed mem intrinsic.

; A memcpy of doubles, typed by "enzyme_type".
define void @memcpy_type(ptr noalias %dst, ptr noalias %src) {
; CHECK-LABEL: @memcpy_type(
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %dst, ptr align 8 %src, i64 256, i1 false), !enzyme_truetype [[DOUBLE:![0-9]+]]
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %sp = getelementptr inbounds double, ptr %src, i64 %i
  %v = load double, ptr %sp, align 8, !enzyme_type !0
  %dp = getelementptr inbounds double, ptr %dst, i64 %i
  store double %v, ptr %dp, align 8
  %inc = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %inc, 32
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; A memset of doubles with an unknown trip count.
define void @memset_truetype(ptr %dst, i64 %n) {
; CHECK-LABEL: @memset_truetype(
; CHECK: call void @llvm.memset.p0.i64(ptr align 8 %dst, i8 0, i64 %{{.*}}, i1 false), !enzyme_truetype [[DOUBLE]]
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %dp = getelementptr inbounds double, ptr %dst, i64 %i
  store double 0.0, ptr %dp, align 8, !enzyme_truetype !2
  %inc = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %inc, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; A {double, integer} layout, repeated over a known size.
define void @memcpy_mixed(ptr noalias %dst, ptr noalias %src) {
; CHECK-LABEL: @memcpy_mixed(
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 16 %dst, ptr align 16 %src, i64 64, i1 false), !enzyme_truetype [[MIXED:![0-9]+]]
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %sp = getelementptr inbounds i128, ptr %src, i64 %i
  %v = load i128, ptr %sp, align 16, !enzyme_truetype !3
  %dp = getelementptr inbounds i128, ptr %dst, i64 %i
  store i128 %v, ptr %dp, align 16, !enzyme_truetype !3
  %inc = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %inc, 4
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; The same with an unknown size: dropped.
define void @memcpy_mixed_unknown(ptr noalias %dst, ptr noalias %src, i64 %n) {
; CHECK-LABEL: @memcpy_mixed_unknown(
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 16 %dst, ptr align 16 %src, i64 %{{.*}}, i1 false){{$}}
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %sp = getelementptr inbounds i128, ptr %src, i64 %i
  %v = load i128, ptr %sp, align 16, !enzyme_truetype !3
  %dp = getelementptr inbounds i128, ptr %dst, i64 %i
  store i128 %v, ptr %dp, align 16
  %inc = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %inc, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; Conflicting annotations: dropped.
define void @memcpy_conflict(ptr noalias %dst, ptr noalias %src) {
; CHECK-LABEL: @memcpy_conflict(
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %dst, ptr align 8 %src, i64 256, i1 false){{$}}
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %sp = getelementptr inbounds i64, ptr %src, i64 %i
  %v = load i64, ptr %sp, align 8, !enzyme_type !0
  %dp = getelementptr inbounds i64, ptr %dst, i64 %i
  store i64 %v, ptr %dp, align 8, !enzyme_truetype !4
  %inc = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %inc, 32
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; CHECK-DAG: [[DOUBLE]] = !{!"Float@double", i64 0}
; CHECK-DAG: [[MIXED]] = !{!"Float@double", i64 0, !"Integer", i64 8, !"Float@double", i64 16, !"Integer", i64 24, !"Float@double", i64 32, !"Integer", i64 40, !"Float@double", i64 48, !"Integer", i64 56}

!0 = !{!"Unknown", i32 -1, !1}
!1 = !{!"Float@double"}
!2 = !{!"Float@double", i64 0}
!3 = !{!"Float@double", i64 0, !"Integer", i64 8}
!4 = !{!"Integer", i64 0}
