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

; The same loop zero-extended. Widening preserves the extension either way, so
; nine significant bits still leave the index a dword. Reading the opcode
; rather than the width the extension produces would price this wide.
; COST-LABEL: 'narrow_zext_index'
; COST: Cost of 10 for VF 8: WIDEN ir<%valB> = load ir<%inB>
; ASM-LABEL: narrow_zext_index:
; ASM-COUNT-1: vpgatherdd
; ASM-NOT: vpgather
define i32 @narrow_zext_index() {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %loop ]
  %inA = getelementptr inbounds [1024 x i8], ptr @A, i64 0, i64 %iv
  %valA = load i8, ptr %inA, align 1
  %ext = zext i8 %valA to i64
  %inB = getelementptr inbounds [1024 x i32], ptr @B, i64 0, i64 %ext
  %valB = load i32, ptr %inB, align 4
  %sum.next = add i32 %sum, %valB
  %next = add nuw nsw i64 %iv, 1
  %done = icmp eq i64 %next, 1024
  br i1 %done, label %exit, label %loop

exit:
  ret i32 %sum.next
}

; A base that an instruction defines but which does not vary: it is loaded once
; in the preheader. The widened GEP has a uniform base and a narrow index, so
; one instruction covers all eight lanes. Reading the base off the scalar GEP
; cannot tell this from the chase below, which is why the recipe reports it.
; COST-LABEL: 'invariant_loaded_base'
; COST: Cost of 10 for VF 8: WIDEN ir<%val> = load ir<%in>
; ASM-LABEL: invariant_loaded_base:
; ASM-COUNT-1: vpgatherdd
; ASM-NOT: vpgather
define i32 @invariant_loaded_base(ptr noalias readonly %pp) {
entry:
  %p = load ptr, ptr %pp, align 8
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %loop ]
  %inA = getelementptr inbounds [1024 x i8], ptr @A, i64 0, i64 %iv
  %valA = load i8, ptr %inA, align 1
  %ext = sext i8 %valA to i64
  %in = getelementptr inbounds i32, ptr %p, i64 %ext
  %val = load i32, ptr %in, align 4
  %sum.next = add i32 %sum, %val
  %next = add nuw nsw i64 %iv, 1
  %done = icmp eq i64 %next, 1024
  br i1 %done, label %exit, label %loop

exit:
  ret i32 %sum.next
}

; A chase with a varying index as well, which is what the narrow index alone
; cannot decide: the base is loaded per iteration, so the widened GEP still has
; a <8 x ptr> base and still needs two instructions. Finding one varying index
; and stopping there would price it as the single dword form.
; COST-LABEL: 'chase_index'
; COST: Cost of 12 for VF 8: WIDEN ir<%v> = load ir<%p>
; ASM-LABEL: chase_index:
; ASM-COUNT-2: vpgatherqd
; ASM-NOT: vpgather
define i32 @chase_index(ptr noalias readonly %bases, ptr noalias readonly %indices) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %loop ]
  %bp = getelementptr inbounds ptr, ptr %bases, i64 %iv
  %base = load ptr, ptr %bp, align 8
  %ip = getelementptr inbounds i8, ptr %indices, i64 %iv
  %raw = load i8, ptr %ip, align 1
  %idx = sext i8 %raw to i64
  %p = getelementptr i32, ptr %base, i64 %idx
  %v = load i32, ptr %p, align 4
  %sum.next = add i32 %sum, %v
  %next = add nuw nsw i64 %iv, 1
  %done = icmp eq i64 %next, 1024
  br i1 %done, label %exit, label %loop

exit:
  ret i32 %sum.next
}
