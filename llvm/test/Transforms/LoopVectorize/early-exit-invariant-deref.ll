; RUN: opt -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -S %s | FileCheck %s

; The VPlan dereferenceability check uses the SCEV overload of
; isDereferenceableAndAlignedInLoop. A loop-invariant address is not an
; add recurrence, so that overload used to reject it even when the
; LoadInst overload proves the access safe.

; The address is loop invariant and dereferenceable for the access.
define i64 @invariant_deref(ptr align 4 dereferenceable(4) %p, i32 %limit) {
; CHECK-LABEL: define i64 @invariant_deref(
; CHECK:       vector.body:
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %ld = load i32, ptr %p, align 4
  %early = icmp eq i32 %ld, %limit
  br i1 %early, label %exit, label %latch

latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %cont = icmp ult i64 %iv.next, 64
  br i1 %cont, label %loop, label %exit

exit:
  %iv.exit = phi i64 [ %iv, %loop ], [ 64, %latch ]
  ret i64 %iv.exit
}

; Constant offset within a dereferenceable object.
define i64 @invariant_offset(ptr align 4 dereferenceable(16) %p, i32 %limit) {
; CHECK-LABEL: define i64 @invariant_offset(
; CHECK:       vector.body:
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %gep = getelementptr inbounds nuw i32, ptr %p, i64 3
  %ld = load i32, ptr %gep, align 4
  %early = icmp eq i32 %ld, %limit
  br i1 %early, label %exit, label %latch

latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %cont = icmp ult i64 %iv.next, 64
  br i1 %cont, label %loop, label %exit

exit:
  %iv.exit = phi i64 [ %iv, %loop ], [ 64, %latch ]
  ret i64 %iv.exit
}

; No dereferenceable information: still rejected.
define i64 @invariant_not_deref(ptr align 4 %p, i32 %limit) {
; CHECK-LABEL: define i64 @invariant_not_deref(
; CHECK-NOT:   vector.body:
; CHECK:       ret i64
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %ld = load i32, ptr %p, align 4
  %early = icmp eq i32 %ld, %limit
  br i1 %early, label %exit, label %latch

latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %cont = icmp ult i64 %iv.next, 64
  br i1 %cont, label %loop, label %exit

exit:
  %iv.exit = phi i64 [ %iv, %loop ], [ 64, %latch ]
  ret i64 %iv.exit
}

; The access starts in range but extends past the object.
define i64 @invariant_past_end(ptr align 4 dereferenceable(16) %p, i32 %limit) {
; CHECK-LABEL: define i64 @invariant_past_end(
; CHECK-NOT:   vector.body:
; CHECK:       ret i64
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %gep = getelementptr inbounds nuw i32, ptr %p, i64 4
  %ld = load i32, ptr %gep, align 4
  %early = icmp eq i32 %ld, %limit
  br i1 %early, label %exit, label %latch

latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %cont = icmp ult i64 %iv.next, 64
  br i1 %cont, label %loop, label %exit

exit:
  %iv.exit = phi i64 [ %iv, %loop ], [ 64, %latch ]
  ret i64 %iv.exit
}

; The object covers the access, but the address is not aligned for it.
define i64 @invariant_misaligned(ptr align 4 dereferenceable(16) %p, i32 %limit) {
; CHECK-LABEL: define i64 @invariant_misaligned(
; CHECK-NOT:   vector.body:
; CHECK:       ret i64
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %gep = getelementptr inbounds nuw i8, ptr %p, i64 1
  %ld = load i32, ptr %gep, align 4
  %early = icmp eq i32 %ld, %limit
  br i1 %early, label %exit, label %latch

latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %cont = icmp ult i64 %iv.next, 64
  br i1 %cont, label %loop, label %exit

exit:
  %iv.exit = phi i64 [ %iv, %loop ], [ 64, %latch ]
  ret i64 %iv.exit
}
