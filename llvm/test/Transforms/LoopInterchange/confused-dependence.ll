; RUN: opt < %s -passes=loop-interchange -pass-remarks-output=%t \
; RUN:     -disable-output
; RUN: FileCheck -input-file %t %s

;; In the following case, p0 and p1 may alias, so the direction vector must be [* *].
;;
;; void may_alias(float *p0, float *p1) {
;;   for (int i = 0; i < 4; i++)
;;     for (int j = 0; j < 4; j++)
;;       p0[4 * i + j] = p1[4 * j + i];
;; }

; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Dependence
; CHECK-NEXT: Function:        may_alias
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          All loops have dependencies in all directions.
; CHECK-NEXT: ...
define void @may_alias(ptr %p0, ptr %p1) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.i.header ], [ %j.inc, %for.j ]
  %idx.0 = getelementptr inbounds [4 x [4 x float]], ptr %p0, i32 0, i32 %j, i32 %i
  %idx.1 = getelementptr inbounds [4 x [4 x float]], ptr %p1, i32 0, i32 %i, i32 %j
  %0 = load float, ptr %idx.0, align 4
  store float %0, ptr %idx.1, align 4
  %j.inc = add nuw nsw i32 %j, 1
  %cmp.j = icmp slt i32 %j.inc, 4
  br i1 %cmp.j, label %for.j, label %for.i.latch

for.i.latch:
  %i.inc = add nuw nsw i32 %i, 1
  %cmp.i = icmp slt i32 %i.inc, 4
  br i1 %cmp.i, label %for.i.header, label %exit

exit:
  ret void
}
