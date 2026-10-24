; RUN: llc < %s -mtriple=s390x-linux-gnu -O2 -stop-after=early-tailduplication \
; RUN:   | FileCheck %s
;
; Computed-goto dispatch: %ig loads the next target from a slot whose address
; depends on a PHI of %ig; l1 and l2 store the next target one slot further
; and branch back. Early tail duplication copies %ig into its predecessors.
; In a copy, the IR value of the load address describes the slot as computed
; by the *next* execution of %ig, so it must not be kept in the memory
; operand: alias analysis would relate it to the store in the same block as
; if both belonged to one execution of %ig and could answer NoAlias although
; both access the same slot. The scheduler could then move the load of the
; branch target above the store (seen on z/OS with GnuCOBOL-generated code).

define i64 @f(ptr %base, ptr %ext) {
; CHECK-LABEL: name: f
; CHECK:       bb.{{[0-9]+}}.l1
; CHECK:         STG {{.*}} :: (store (s64) into %ir.
; CHECK-NEXT:    {{.*}} = LA
; CHECK-NEXT:    {{.*}} = LG {{.*}} :: (load (s64))
; CHECK:       bb.{{[0-9]+}}.l2
; CHECK:         STG {{.*}} :: (store (s64) into %ir.
; CHECK-NEXT:    {{.*}} = LA
; CHECK-NEXT:    {{.*}} = LG {{.*}} :: (load (s64))
entry:
  store ptr blockaddress(@f, %l1), ptr %base
  br label %ig

ig:
  %idx = phi i64 [ 0, %entry ], [ %idx1, %l1 ], [ %idx2, %l2 ]
  %p = getelementptr i8, ptr %base, i64 %idx
  %t = load ptr, ptr %p, align 8
  indirectbr ptr %t, [label %l1, label %l2, label %done]

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
