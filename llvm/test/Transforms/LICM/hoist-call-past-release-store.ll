; RUN: opt -passes='loop-mssa(licm)' -S < %s | FileCheck %s
;
; A read-only call, like a load, may be hoisted above a release store to
; memory it does not access: the release ordering constrains only
; program-order-earlier accesses. Keep seq_cst stores and release RMWs as
; conservative barriers.

@flag = global i32 0
@data = global i32 0

declare i32 @read_data(ptr) nounwind willreturn memory(argmem: read)

define i32 @hoist_call_past_release_store(i32 %n) {
; CHECK-LABEL: @hoist_call_past_release_store(
; CHECK:       entry:
; CHECK-NEXT:    [[V:%.*]] = call i32 @read_data(ptr @data)
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK-NOT:     call i32 @read_data
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %loop ]
  %v = call i32 @read_data(ptr @data)
  %sum.next = add i32 %sum, %v
  store atomic i32 1, ptr @flag release, align 4
  %i.next = add nuw nsw i32 %i, 1
  %cmp = icmp slt i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret i32 %sum.next
}

; Negative: seq_cst stores stay conservative.
define i32 @no_hoist_call_past_seq_cst_store(i32 %n) {
; CHECK-LABEL: @no_hoist_call_past_seq_cst_store(
; CHECK:       loop:
; CHECK:         call i32 @read_data(ptr @data)
; CHECK-NEXT:    {{%.*}} = add
; CHECK-NEXT:    store atomic i32 1, ptr @flag seq_cst
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %loop ]
  %v = call i32 @read_data(ptr @data)
  %sum.next = add i32 %sum, %v
  store atomic i32 1, ptr @flag seq_cst, align 4
  %i.next = add nuw nsw i32 %i, 1
  %cmp = icmp slt i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret i32 %sum.next
}

; Negative: the call may read the location the release store writes; that is
; a real data dependence, not just ordering.
define i32 @no_hoist_call_reads_flag(i32 %n) {
; CHECK-LABEL: @no_hoist_call_reads_flag(
; CHECK:       loop:
; CHECK:         call i32 @read_data(ptr @flag)
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %loop ]
  %v = call i32 @read_data(ptr @flag)
  %sum.next = add i32 %sum, %v
  store atomic i32 1, ptr @flag release, align 4
  %i.next = add nuw nsw i32 %i, 1
  %cmp = icmp slt i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret i32 %sum.next
}
