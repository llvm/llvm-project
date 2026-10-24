; RUN: opt -passes='print<access-info>' -disable-output %s 2>&1 | FileCheck %s

; The unannotated scratch store cannot alias the annotated accesses, which
; need no runtime checks.
; CHECK-LABEL: 'parallel_with_unannotated_alloca'
; CHECK-NEXT:    for.body:
; CHECK-NEXT:      Memory dependences are safe
; CHECK-NEXT:      Dependences:
; CHECK-NEXT:      Run-time memory checks:
; CHECK-NEXT:      Grouped accesses:
; CHECK-EMPTY:
define void @parallel_with_unannotated_alloca(ptr %a, ptr %b, i64 %n) {
entry:
  %scratch = alloca [4 x i32]
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  %v = load i32, ptr %gep.a, align 4, !llvm.access.group !1
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
  store i32 %v, ptr %gep.b, align 4, !llvm.access.group !1
  %idx = and i64 %iv, 3
  %gep.scratch = getelementptr inbounds [4 x i32], ptr %scratch, i64 0, i64 %idx
  store i32 %v, ptr %gep.scratch, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %exit, label %for.body, !llvm.loop !0

exit:
  ret void
}

; Same as above, but the load from %a is not annotated. %a and %b must still be
; checked at run time.
;
; CHECK-LABEL: 'unannotated_load'
; CHECK-NEXT:    for.body:
; CHECK-NEXT:      Memory dependences are safe with run-time checks
; CHECK-NEXT:      Dependences:
; CHECK-NEXT:      Run-time memory checks:
; CHECK-NEXT:      Check 0:
; CHECK-NEXT:        Comparing group GRP0:
; CHECK-NEXT:          %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
; CHECK-NEXT:        Against group GRP1:
; CHECK-NEXT:          %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
define void @unannotated_load(ptr %a, ptr %b, i64 %n) {
entry:
  %scratch = alloca [4 x i32]
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  %v = load i32, ptr %gep.a, align 4
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
  store i32 %v, ptr %gep.b, align 4, !llvm.access.group !1
  %idx = and i64 %iv, 3
  %gep.scratch = getelementptr inbounds [4 x i32], ptr %scratch, i64 0, i64 %idx
  store i32 %v, ptr %gep.scratch, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %exit, label %for.body, !llvm.loop !0

exit:
  ret void
}

; The pointer %gep.a is loaded with an annotation but also stored to without
; one. Any pointer with an unannotated access must be treated as not parallel.
;
; CHECK-LABEL: 'same_pointer_mixed_annotations'
; CHECK-NEXT:    for.body:
; CHECK-NEXT:      Memory dependences are safe with run-time checks
; CHECK-NEXT:      Dependences:
; CHECK-NEXT:        Forward:
; CHECK-NEXT:            %v = load i32, ptr %gep.a, align 4, !llvm.access.group !{{[0-9]+}} ->
; CHECK-NEXT:            store i32 %v2, ptr %gep.a, align 4
; CHECK-EMPTY:
; CHECK-NEXT:      Run-time memory checks:
; CHECK-NEXT:      Check 0:
; CHECK-NEXT:        Comparing group GRP0:
; CHECK-NEXT:          %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
; CHECK-NEXT:        Against group GRP1:
; CHECK-NEXT:          %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
define void @same_pointer_mixed_annotations(ptr %a, ptr %b, i64 %n) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  %v = load i32, ptr %gep.a, align 4, !llvm.access.group !1
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
  store i32 %v, ptr %gep.b, align 4, !llvm.access.group !1
  %v2 = add i32 %v, 1
  store i32 %v2, ptr %gep.a, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %exit, label %for.body, !llvm.loop !0

exit:
  ret void
}

; A loop-carried dependence through unannotated scratch accesses must still
; prevent vectorization.
;
; CHECK-LABEL: 'unannotated_alloca_with_dependence'
; CHECK-NEXT:    for.body:
; CHECK-NEXT:      Report: unsafe dependent memory operations in loop.
; CHECK-NEXT:      Backward loop carried data dependence.
; CHECK-NEXT:      Dependences:
; CHECK-NEXT:        Backward:
; CHECK-NEXT:            %s = load i32, ptr %gep.scratch, align 4 ->
; CHECK-NEXT:            store i32 %sum, ptr %gep.scratch.next, align 4
; CHECK-EMPTY:
; CHECK-NEXT:      Run-time memory checks:
; CHECK-NEXT:      Grouped accesses:
; CHECK-EMPTY:
define void @unannotated_alloca_with_dependence(ptr %a, ptr %b, i64 %n) {
entry:
  %scratch = alloca [1024 x i32]
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  %v = load i32, ptr %gep.a, align 4, !llvm.access.group !1
  %gep.scratch = getelementptr inbounds [1024 x i32], ptr %scratch, i64 0, i64 %iv
  %s = load i32, ptr %gep.scratch, align 4
  %sum = add i32 %v, %s
  %iv.next = add nuw nsw i64 %iv, 1
  %gep.scratch.next = getelementptr inbounds [1024 x i32], ptr %scratch, i64 0, i64 %iv.next
  store i32 %sum, ptr %gep.scratch.next, align 4
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
  store i32 %sum, ptr %gep.b, align 4, !llvm.access.group !1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %exit, label %for.body, !llvm.loop !0

exit:
  ret void
}

; Accesses carrying an access group that is *not* listed in the loop's
; llvm.loop.parallel_accesses are not parallel with respect to this loop.
;
; CHECK-LABEL: 'access_group_not_in_loop'
; CHECK-NEXT:    for.body:
; CHECK-NEXT:      Memory dependences are safe with run-time checks
; CHECK-NEXT:      Dependences:
; CHECK-NEXT:      Run-time memory checks:
; CHECK-NEXT:      Check 0:
define void @access_group_not_in_loop(ptr %a, ptr %b, i64 %n) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  %v = load i32, ptr %gep.a, align 4, !llvm.access.group !2
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
  store i32 %v, ptr %gep.b, align 4, !llvm.access.group !2
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %exit, label %for.body, !llvm.loop !0

exit:
  ret void
}

; Different groups listed by the loop, including a group in a list, exempt
; even accesses to the same underlying object from dependence checking.
; CHECK-LABEL: 'parallel_access_group_list'
; CHECK: Memory dependences are safe
; CHECK-NEXT: Dependences:
; CHECK-NEXT: Run-time memory checks:
; CHECK-NEXT: Grouped accesses:
; CHECK-EMPTY:
define void @parallel_access_group_list(ptr %a, i64 %n) {
entry:
  %scratch = alloca i32
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  %v = load i32, ptr %gep.a, align 4, !llvm.access.group !3
  %iv.next = add nuw nsw i64 %iv, 1
  %gep.next = getelementptr inbounds i32, ptr %a, i64 %iv.next
  store i32 %v, ptr %gep.next, align 4, !llvm.access.group !4
  store i32 %v, ptr %scratch, align 4
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %exit, label %for.body, !llvm.loop !5

exit:
  ret void
}

; A second store to an already registered pointer must still prevent skipping
; its alias set if the second store is unannotated.
; CHECK-LABEL: 'duplicate_store_mixed_annotations'
; CHECK: Memory dependences are safe with run-time checks
; CHECK: Run-time memory checks:
; CHECK-NEXT: Check 0:
define void @duplicate_store_mixed_annotations(ptr %a, ptr %b, i64 %n) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  %v = load i32, ptr %gep.a, align 4, !llvm.access.group !1
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
  store i32 %v, ptr %gep.b, align 4, !llvm.access.group !1
  store i32 0, ptr %gep.b, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %exit, label %for.body, !llvm.loop !0

exit:
  ret void
}

; An unannotated access through a different pointer in the same alias set can
; carry a dependence with an annotated access.
; CHECK-LABEL: 'mixed_alias_set_with_dependence'
; CHECK: Report: unsafe dependent memory operations in loop.
; CHECK-NEXT: Backward loop carried data dependence.
define void @mixed_alias_set_with_dependence(ptr %a, i64 %n) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  %v = load i32, ptr %gep.a, align 4, !llvm.access.group !1
  %iv.next = add nuw nsw i64 %iv, 1
  %gep.next = getelementptr inbounds i32, ptr %a, i64 %iv.next
  store i32 %v, ptr %gep.next, align 4
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %exit, label %for.body, !llvm.loop !0

exit:
  ret void
}

; An unannotated access through a non-header PHI must mark both incoming
; pointers as unannotated, even if they also have annotated accesses.
; CHECK-LABEL: 'unannotated_forked_pointer'
; CHECK: Memory dependences are safe with run-time checks
; CHECK: Run-time memory checks:
; CHECK-NEXT: Check 0:
define void @unannotated_forked_pointer(ptr %a, ptr %b, ptr %c, i64 %n, i1 %cond) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %merge ]
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
  %gep.c = getelementptr inbounds i32, ptr %c, i64 %iv
  %v = load i32, ptr %gep.c, align 4, !llvm.access.group !1
  store i32 %v, ptr %gep.a, align 4, !llvm.access.group !1
  store i32 %v, ptr %gep.b, align 4, !llvm.access.group !1
  br i1 %cond, label %then, label %merge

then:
  br label %merge

merge:
  %ptr = phi ptr [ %gep.a, %for.body ], [ %gep.b, %then ]
  store i32 0, ptr %ptr, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %exit, label %for.body, !llvm.loop !0

exit:
  ret void
}

; Legacy annotations also work with an unrelated unannotated access. Cover
; both a direct loop ID and a list containing that ID.
; CHECK-LABEL: 'legacy_parallel_accesses'
; CHECK: Memory dependences are safe
; CHECK-NEXT: Dependences:
; CHECK-NEXT: Run-time memory checks:
; CHECK-NEXT: Grouped accesses:
; CHECK-EMPTY:
define void @legacy_parallel_accesses(ptr %a, ptr %b, i64 %n) {
entry:
  %scratch = alloca i32
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  %v = load i32, ptr %gep.a, align 4, !llvm.access.group !2, !llvm.mem.parallel_loop_access !6
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
  store i32 %v, ptr %gep.b, align 4, !llvm.mem.parallel_loop_access !7
  store i32 %v, ptr %scratch, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %exit, label %for.body, !llvm.loop !6

exit:
  ret void
}

!0 = distinct !{!0, !{!"llvm.loop.parallel_accesses", !1}}
!1 = distinct !{}
!2 = distinct !{}
!3 = !{!2, !1}
!4 = distinct !{}
!5 = distinct !{!5, !{!"llvm.loop.parallel_accesses", !1, !4}}
!6 = distinct !{!6}
!7 = !{!0, !6}
