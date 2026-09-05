; RUN: opt -passes=instcombine -S < %s | FileCheck %s

; The store mergeStoreIntoSuccessor creates must keep the access groups both
; original stores agree on, or the loop stops being annotated-parallel.

; CHECK-LABEL: @merged_store_keeps_shared_access_group(
; CHECK: store double %storemerge, ptr %q, align 8, !llvm.access.group [[ACC:![0-9]+]]
define void @merged_store_keeps_shared_access_group(ptr noalias %dst, ptr noalias %src, i64 %n) {
entry:
  %guard = icmp sgt i64 %n, 0
  br i1 %guard, label %loop, label %exit

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  %p = getelementptr inbounds double, ptr %src, i64 %i
  %v = load double, ptr %p, align 8, !llvm.access.group !0
  %q = getelementptr inbounds double, ptr %dst, i64 %i
  %c = fcmp ogt double %v, 0.000000e+00
  br i1 %c, label %then, label %else

then:
  store double %v, ptr %q, align 8, !llvm.access.group !0
  br label %latch

else:
  store double 1.000000e+00, ptr %q, align 8, !llvm.access.group !0
  br label %latch

latch:
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop, !llvm.loop !1

exit:
  ret void
}

; If only one store is in the group, the merged store must not claim it.

; CHECK-LABEL: @merged_store_drops_unshared_access_group(
; CHECK: store double %storemerge, ptr %q, align 8
; CHECK-NOT: !llvm.access.group
define void @merged_store_drops_unshared_access_group(ptr noalias %dst, ptr noalias %src, i64 %n) {
entry:
  %guard = icmp sgt i64 %n, 0
  br i1 %guard, label %loop, label %exit

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  %p = getelementptr inbounds double, ptr %src, i64 %i
  %v = load double, ptr %p, align 8
  %q = getelementptr inbounds double, ptr %dst, i64 %i
  %c = fcmp ogt double %v, 0.000000e+00
  br i1 %c, label %then, label %else

then:
  store double %v, ptr %q, align 8, !llvm.access.group !0
  br label %latch

else:
  store double 1.000000e+00, ptr %q, align 8
  br label %latch

latch:
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; CHECK: [[ACC]] = distinct !{}

!0 = distinct !{}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.parallel_accesses", !0}
