; Pins the index width a gather is costed at against the width CodeGen gives
; it, for the consumer that asks before the address has been widened.
;
; LoopVectorize queries the cost of a widened load with the original scalar
; address, where the component that varies across lanes is not spelled out: a
; pointer chase and a uniform base reached through a narrow index both present
; a GEP whose operands are scalars. The two lower differently, so each cost
; check below is paired with the gather count llc emits for the same loop.
;
; These checks are maintained by hand rather than by
; update_analyze_test_checks.py, which does not know about the paired llc run.

; RUN: opt -passes=loop-vectorize -force-vector-width=8 -force-vector-interleave=1 -mtriple=x86_64-unknown-linux-gnu -mcpu=skylake -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=COST
; RUN: opt -passes=loop-vectorize -force-vector-width=8 -force-vector-interleave=1 -mtriple=x86_64-unknown-linux-gnu -mcpu=skylake -S < %s -o %t.ll
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mcpu=skylake %t.ll -o - | FileCheck %s --check-prefix=ASM

; REQUIRES: asserts

%S = type { i32, i32 }

@A = global [1024 x i8] zeroinitializer, align 128
@B = global [1024 x i32] zeroinitializer, align 128

; The base is loaded on each iteration and the field indices are constants, so
; nothing in the index list varies across lanes and the widened form gathers
; from a vector of pointers. That needs pointer-width indices, which hold half
; as many lanes per register, so eight lanes take two instructions. Reading the
; index list alone would find no varying index and price this as a dword form
; covering all eight in one.
; COST-LABEL: 'chase'
; COST: Cost of 12 for VF 8: WIDEN ir<%y> = load ir<%field>
; ASM-LABEL: chase:
; ASM-COUNT-2: vpgatherqd
; ASM-NOT: vpgather
define i32 @chase(ptr noalias readonly %p) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %loop ]
  %pi = getelementptr inbounds ptr, ptr %p, i64 %iv
  %q = load ptr, ptr %pi, align 8
  %field = getelementptr inbounds %S, ptr %q, i64 0, i32 1
  %y = load i32, ptr %field, align 4
  %sum.next = add i32 %sum, %y
  %next = add nuw nsw i64 %iv, 1
  %done = icmp eq i64 %next, 1024
  br i1 %done, label %exit, label %loop

exit:
  ret i32 %sum.next
}

; The control, and the reason the case above cannot simply be assumed wide: the
; base is a global and it is the index that varies, sign-extended from a byte
; and so narrow enough to stay a dword. One instruction covers all eight lanes.
; COST-LABEL: 'narrow_index'
; COST: Cost of 10 for VF 8: WIDEN ir<%valB> = load ir<%inB>
; ASM-LABEL: narrow_index:
; ASM-COUNT-1: vpgatherdd
; ASM-NOT: vpgather
define i32 @narrow_index() {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %loop ]
  %inA = getelementptr inbounds [1024 x i8], ptr @A, i64 0, i64 %iv
  %valA = load i8, ptr %inA, align 1
  %ext = sext i8 %valA to i64
  %inB = getelementptr inbounds [1024 x i32], ptr @B, i64 0, i64 %ext
  %valB = load i32, ptr %inB, align 4
  %sum.next = add i32 %sum, %valB
  %next = add nuw nsw i64 %iv, 1
  %done = icmp eq i64 %next, 1024
  br i1 %done, label %exit, label %loop

exit:
  ret i32 %sum.next
}
