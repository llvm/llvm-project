; RUN: llc -mtriple=x86_64-- -o /dev/null %s

; The bitcasts around the loop carried PHI used to make the rewrite map built
; by the peephole optimizer cyclic, which sent getNewSource() into infinite
; recursion (issue #36621). Check that this compiles at all.
;
; Note that the unused %sel and %cmp both look dead but are required to get
; instruction selection to produce the cross register bank bitcast pattern
; that triggered the bug.

define void @phi_cycle(double %x, i1 %c) {
top:
  %i = bitcast double %x to i64
  %sel = select i1 false, i64 %i, i64 0
  br label %loop

loop:
  %phi = phi i64 [ %i, %top ], [ %back, %latch ]
  %d = bitcast i64 %phi to double
  br i1 %c, label %exit, label %latch

exit:
  unreachable

latch:
  %cmp = fcmp ule double 0.000000e+00, %d
  %back = bitcast double %d to i64
  br label %loop
}
