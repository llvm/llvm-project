; RUN: llc -mtriple=hexagon -stop-after=expand-condsets -o - %s | FileCheck %s

; Recalculating live intervals after expanding the select must preserve the
; low subregister's live-in segment even though a later partial definition
; makes that subregister undefined.

define i64 @f(ptr %unused0, i32 %a, i8 %unused1, i4 %sel, i1 %b, i64 %x) {
  ; CHECK-LABEL: name: f
  ; CHECK:       [[PAIR:%[0-9]+]].isub_lo:doubleregs = COPY
  ; CHECK:       bb.3.bb1:
  ; CHECK:       [[PRED:%[0-9]+]]:predregs = S2_tstbit_i
  ; CHECK-NEXT:  [[VALUE:%[0-9]+]]:intregs = A2_paddit [[PRED]], [[PAIR]].isub_lo, 1
  ; CHECK-NEXT:  [[VALUE]]:intregs = A2_tfrf [[PRED]], [[PAIR]].isub_lo, implicit [[VALUE]](tied-def 0)
  ; CHECK:       [[PAIR]].isub_hi:doubleregs = A2_tfrsi 26
entry:
  switch i4 %sel, label %exit [
    i4 -4, label %bb2
    i4 -5, label %bb1
  ]

bb1:
  %p = phi i32 [ %a, %bb2 ], [ 0, %entry ]
  %z = zext i1 %b to i32
  %add = add i32 %p, %z
  %cmp = icmp ugt i32 %add, 0
  br i1 %cmp, label %exit, label %bb3

bb2:
  br label %bb1

bb3:
  %cmp2 = icmp eq i32 %a, 0
  br i1 %cmp2, label %bb4, label %exit

bb4:
  %lo = trunc i64 %x to i32
  %sh = lshr i64 %x, 32
  %hi = trunc i64 %sh to i16
  br label %exit

exit:
  %v1 = phi i16 [ 0, %entry ], [ 0, %bb4 ], [ 1, %bb1 ], [ 0, %bb3 ]
  %v2 = phi i16 [ 1, %entry ], [ %hi, %bb4 ], [ 26, %bb1 ], [ 26, %bb3 ]
  %v3 = phi i32 [ 0, %entry ], [ %lo, %bb4 ], [ 0, %bb1 ], [ poison, %bb3 ]
  %e2 = zext i16 %v2 to i64
  %e1 = zext i16 %v1 to i64
  %o1 = or i64 %e2, %e1
  %e3 = zext i32 %v3 to i64
  %r = or i64 %o1, %e3
  ret i64 %r
}
