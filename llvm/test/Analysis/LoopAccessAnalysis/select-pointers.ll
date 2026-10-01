; RUN: opt -passes='print<access-info>' -disable-output %s 2>&1 | FileCheck %s

; Loads through a select between two pointers are analyzed as separate
; accesses through each operand, like non-header pointer phis.

; q[i] = c[i] ? p[i] : q[i]. The read of q[i] is a forward dependence to the
; store of q[i] in the same iteration, p[i] and q[i] do not alias.
define void @select_ptr_load_stored_back(ptr noalias %p, ptr noalias %q, ptr noalias %c, i64 %n) {
; CHECK-LABEL: 'select_ptr_load_stored_back'
; CHECK-NEXT:    loop:
; CHECK-NEXT:      Memory dependences are safe
; CHECK-NEXT:      Dependences:
; CHECK-NEXT:        Forward:
; CHECK-NEXT:            %v = load float, ptr %sel, align 4 ->
; CHECK-NEXT:            store float %v, ptr %gep.q, align 4
; CHECK-EMPTY:
; CHECK-NEXT:      Run-time memory checks:
; CHECK-NEXT:      Grouped accesses:
; CHECK-EMPTY:
; CHECK-NEXT:      Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:      SCEV assumptions:
; CHECK-EMPTY:
; CHECK-NEXT:      Expressions re-written:
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.c = getelementptr inbounds float, ptr %c, i64 %iv
  %cv = load float, ptr %gep.c, align 4
  %cond = fcmp une float %cv, 0.000000e+00
  %gep.p = getelementptr inbounds float, ptr %p, i64 %iv
  %gep.q = getelementptr inbounds float, ptr %q, i64 %iv
  %sel = select i1 %cond, ptr %gep.p, ptr %gep.q
  %v = load float, ptr %sel, align 4
  store float %v, ptr %gep.q, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

; Without noalias, the p[i] read needs a run-time check against the q[i] store,
; while the q[i] read is still a forward dependence.
define void @select_ptr_load_stored_back_may_alias(ptr %p, ptr %q, ptr noalias %c, i64 %n) {
; CHECK-LABEL: 'select_ptr_load_stored_back_may_alias'
; CHECK-NEXT:    loop:
; CHECK-NEXT:      Memory dependences are safe with run-time checks
; CHECK-NEXT:      Dependences:
; CHECK-NEXT:        Forward:
; CHECK-NEXT:            %v = load float, ptr %sel, align 4 ->
; CHECK-NEXT:            store float %v, ptr %gep.q, align 4
; CHECK-EMPTY:
; CHECK-NEXT:      Run-time memory checks:
; CHECK-NEXT:      Check 0:
; CHECK-NEXT:        Comparing group GRP0:
; CHECK-NEXT:          %gep.q = getelementptr inbounds float, ptr %q, i64 %iv
; CHECK-NEXT:          %gep.q = getelementptr inbounds float, ptr %q, i64 %iv
; CHECK-NEXT:        Against group GRP1:
; CHECK-NEXT:          %gep.p = getelementptr inbounds float, ptr %p, i64 %iv
; CHECK-NEXT:      Grouped accesses:
; CHECK-NEXT:        Group GRP0:
; CHECK-NEXT:          (Low: %q High: ((4 * %n) + %q))
; CHECK-NEXT:            Member: {%q,+,4}<nuw><%loop>
; CHECK-NEXT:            Member: {%q,+,4}<nuw><%loop>
; CHECK-NEXT:        Group GRP1:
; CHECK-NEXT:          (Low: %p High: ((4 * %n) + %p))
; CHECK-NEXT:            Member: {%p,+,4}<nw><%loop>
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.c = getelementptr inbounds float, ptr %c, i64 %iv
  %cv = load float, ptr %gep.c, align 4
  %cond = fcmp une float %cv, 0.000000e+00
  %gep.p = getelementptr inbounds float, ptr %p, i64 %iv
  %gep.q = getelementptr inbounds float, ptr %q, i64 %iv
  %sel = select i1 %cond, ptr %gep.p, ptr %gep.q
  %v = load float, ptr %sel, align 4
  store float %v, ptr %gep.q, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

; A store through a select of pointers writes to one of two strided locations
; and is checked against each of them: p[i+1] may be read in the next iteration.
define void @select_ptr_store_backward(ptr noalias %p, ptr noalias %c, i64 %n) {
; CHECK-LABEL: 'select_ptr_store_backward'
; CHECK-NEXT:    loop:
; CHECK-NEXT:      Report: unsafe dependent memory operations in loop.
; CHECK-NEXT:  Backward loop carried data dependence.
; CHECK-NEXT:      Dependences:
; CHECK-NEXT:        Forward:
; CHECK-NEXT:            %v = load float, ptr %gep.p, align 4 ->
; CHECK-NEXT:            store float %v, ptr %sel, align 4
; CHECK-EMPTY:
; CHECK-NEXT:        Backward:
; CHECK-NEXT:            %v = load float, ptr %gep.p, align 4 ->
; CHECK-NEXT:            store float %v, ptr %sel, align 4
; CHECK-EMPTY:
; CHECK-NEXT:        Forward:
; CHECK-NEXT:            store float %v, ptr %sel, align 4 ->
; CHECK-NEXT:            store float %v, ptr %sel, align 4
; CHECK-EMPTY:
; CHECK-NEXT:      Run-time memory checks:
; CHECK-NEXT:      Grouped accesses:
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.c = getelementptr inbounds float, ptr %c, i64 %iv
  %cv = load float, ptr %gep.c, align 4
  %cond = fcmp une float %cv, 0.000000e+00
  %gep.p = getelementptr inbounds float, ptr %p, i64 %iv
  %v = load float, ptr %gep.p, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %gep.p.next = getelementptr inbounds float, ptr %p, i64 %iv.next
  %sel = select i1 %cond, ptr %gep.p, ptr %gep.p.next
  store float %v, ptr %sel, align 4
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}
