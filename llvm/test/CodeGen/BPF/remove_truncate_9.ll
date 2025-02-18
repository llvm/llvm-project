; RUN: llc -mcpu=v2 -mtriple=bpf < %s | FileCheck %s
; RUN: llc -mcpu=v4 -mtriple=bpf < %s | FileCheck %s

; Zero extension instructions should be eliminated at instruction
; selection phase for all test cases below.

; In BPF zero extension is implemented as &= or a pair of <<=/>>=
; instructions, hence simply check that &= and >>= do not exist in
; generated code (<<= remains because %c is used by both call and
; lshr in a few test cases).

; CHECK-NOT: &=
; CHECK-NOT: >>=

define void @shl_lshr_same_bb(ptr %p) {
entry:
  %a = load i8, ptr %p, align 1
  %b = zext i8 %a to i64
  %c = shl i64 %b, 56
  %d = lshr i64 %c, 56
  %e = icmp eq i64 %d, 0
  ; hasOneUse() is a common requirement for many CombineDAG
  ; transofmations, make sure that it does not matter in this case.
  call void @sink1(i8 %a, i64 %b, i64 %c, i64 %d, i1 %e)
  ret void
}

define void @shl_lshr_diff_bb(ptr %p) {
entry:
  %a = load i16, ptr %p, align 2
  %b = zext i16 %a to i64
  %c = shl i64 %b, 48
  %d = lshr i64 %c, 48
  br label %next

; Jump to the new basic block creates a COPY instruction for %d, which
; might be materialized as noop or as AND_ri (zero extension) at the
; start of the basic block. The decision depends on TLI.isZExtFree()
; results, see RegsForValue::getCopyToRegs(). Check below verifies
; that COPY is materialized as noop.
next:
  %e = icmp eq i64 %d, 0
  call void @sink2(i16 %a, i64 %b, i64 %c, i64 %d, i1 %e)
  ret void
}

define void @load_zext_same_bb(ptr %p) {
entry:
  %a = load i8, ptr %p, align 1
  ; zext is implicit in this context
  %b = icmp eq i8 %a, 0
  call void @sink3(i8 %a, i1 %b)
  ret void
}

define void @load_zext_diff_bb(ptr %p) {
entry:
  %a = load i8, ptr %p, align 1
  br label %next

next:
  %b = icmp eq i8 %a, 0
  call void @sink3(i8 %a, i1 %b)
  ret void
}

define void @load_zext_diff_bb_2(ptr %p) {
entry:
  %a = load i32, ptr %p, align 4
  br label %next

next:
  %b = icmp eq i32 %a, 0
  call void @sink4(i32 %a, i1 %b)
  ret void
}

declare void @sink1(i8, i64, i64, i64, i1);
declare void @sink2(i16, i64, i64, i64, i1);
declare void @sink3(i8, i1);
declare void @sink4(i32, i1);
