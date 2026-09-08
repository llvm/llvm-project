; RUN: opt -passes='loop(indvars)' -S < %s | FileCheck %s --check-prefix=INDVARS
; RUN: opt -disable-output -passes='print<scalar-evolution>' < %s 2>&1 | FileCheck %s --check-prefix=SCEV

; The latch compare is a samesign unsigned comparison against a positive
; constant, so for every well-defined execution %remaining is non-negative.
; Deriving the exit limit from the equivalent signed predicate bounds the trip
; count by i32 signed range (2^24 - 1 for a step of 128) instead of the
; unsigned one (2^25 - 1), which is exactly what allows the parallel
; offset IV to be proven non-wrapping.

declare void @use(i32)

define void @countdown_with_offset(i32 %remaining.start) {
; INDVARS-LABEL: @countdown_with_offset(
; INDVARS:       loop:
; INDVARS:         %offset.next = add nuw i32 %offset, 128
;
; SCEV-LABEL: Determining loop execution counts for: @countdown_with_offset
; SCEV-NEXT:  Loop %loop: backedge-taken count is ((127 + (-1 * (128 smin %remaining.start)) + %remaining.start) /u 128)
; SCEV-NEXT:  Loop %loop: constant max backedge-taken count is i32 16777215
entry:
  br label %loop

loop:
  %remaining = phi i32 [ %remaining.start, %entry ], [ %remaining.next, %loop ]
  %offset = phi i32 [ 0, %entry ], [ %offset.next, %loop ]
  call void @use(i32 %offset)
  %remaining.next = add i32 %remaining, -128
  %offset.next = add i32 %offset, 128
  %continue = icmp samesign ugt i32 %remaining, 128
  br i1 %continue, label %loop, label %exit

exit:
  ret void
}
