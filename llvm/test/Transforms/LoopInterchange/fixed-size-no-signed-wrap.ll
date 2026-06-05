; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 \
; RUN:     -pass-remarks-output=%t -disable-output
; RUN: FileCheck --input-file=%t %s

; The inner array dimension is indexed by a zero-extended i32 counter whose
; loop is governed by a separate countdown, so scalar evolution cannot attach
; <nsw> to the delinearized subscript. Fixed-size delinearization proves the
; subscript stays within the array bound, so dependence analysis can prove the
; recurrence does not wrap and the loops are interchanged. This mirrors a
; column-major A(5, 5) access A(i, j) = A(i, j) + 1.

; CHECK:       --- !Passed
; CHECK-NEXT:  Pass:            loop-interchange
; CHECK-NEXT:  Name:            Interchanged
; CHECK-NEXT:  Function:        fixed_size_5x5
; CHECK-NEXT:  Args:
; CHECK-NEXT:    - String:          Loop interchanged with enclosing loop.

;   real :: A(5, 5)
;   do i = 1, 5
;     do j = 1, 5
;       A(i, j) = A(i, j) + 1.0
;     end do
;   end do
define void @fixed_size_5x5(ptr noalias %A) {
entry:
  br label %outer.header

outer.header:
  %i.count = phi i64 [ 5, %entry ], [ %i.count.next, %outer.latch ]
  %i = phi i32 [ 1, %entry ], [ %i.next, %outer.latch ]
  %i.ext = zext nneg i32 %i to i64
  %row.gep = getelementptr [4 x i8], ptr %A, i64 %i.ext
  br label %inner

inner:
  %j = phi i64 [ 1, %outer.header ], [ %j.next, %inner ]
  %j.count = phi i64 [ 5, %outer.header ], [ %j.count.next, %inner ]
  %col.off = mul nuw nsw i64 %j, 20
  %elt.gep = getelementptr i8, ptr %row.gep, i64 %col.off
  %addr = getelementptr i8, ptr %elt.gep, i64 -24
  %v = load float, ptr %addr, align 4
  %inc = fadd contract float %v, 1.000000e+00
  store float %inc, ptr %addr, align 4
  %j.next = add nuw nsw i64 %j, 1
  %j.count.next = add nsw i64 %j.count, -1
  %j.done = icmp eq i64 %j.count.next, 0
  br i1 %j.done, label %outer.latch, label %inner

outer.latch:
  %i.next = add nuw nsw i32 %i, 1
  %i.count.next = add nsw i64 %i.count, -1
  %i.cmp = icmp sgt i64 %i.count, 1
  br i1 %i.cmp, label %outer.header, label %exit

exit:
  ret void
}
