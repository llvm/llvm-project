; RUN: llc < %s -O2 -mtriple=s390x-linux-gnu -debug-only=machine-scheduler \
; RUN:   -o /dev/null 2>&1 | FileCheck %s
; REQUIRES: asserts
;
; Early tail duplication copies the load of the computed goto target (address
; computed from the PHI %idx) into the predecessors. In the copy, the memory
; operand still refers to %p, but %idx is now the value of the next execution
; of %ig, i.e. the slot that %l1 has just stored (%q1). Alias analysis must not
; assume that %idx is the same value in both memory operands (cross-iteration
; mode), otherwise the scheduler sees no dependency and may move the load above
; the store.

define i64 @f(ptr %base, ptr %ext) {
entry:
  store ptr blockaddress(@f, %l1), ptr %base
  br label %ig

ig:
  %idx = phi i64 [ 0, %entry ], [ %idx1, %l1 ], [ %idx2, %l2 ]
  %p = getelementptr i8, ptr %base, i64 %idx
  %t = load ptr, ptr %p, align 8
  indirectbr ptr %t, [label %l1, label %l2, label %done]

; CHECK-LABEL: f:%bb.1 l1
; CHECK:       SU([[ST:[0-9]+]]): STG {{.*}} :: (store (s64) into %ir.scevgep{{[0-9]+}})
; CHECK:       SU({{[0-9]+}}): {{.*}} = LG {{.*}} :: (load (s64) from %ir.scevgep{{[0-9]+}})
; CHECK:       Predecessors:
; CHECK-NOT:   Successors:
; CHECK:       SU([[ST]]): Ord Latency=1 Memory
l1:
  store volatile i64 1, ptr %ext
  %idx1 = add nsw i64 %idx, 16
  %q1 = getelementptr inbounds i8, ptr %base, i64 %idx1
  store ptr blockaddress(@f, %l2), ptr %q1, align 8
  br label %ig

l2:
  store volatile i64 2, ptr %ext
  %idx2 = add nsw i64 %idx, 16
  %q2 = getelementptr inbounds i8, ptr %base, i64 %idx2
  store ptr blockaddress(@f, %done), ptr %q2, align 8
  br label %ig

done:
  ret i64 %idx
}
