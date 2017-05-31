; RUN: opt < %s -loop-unroll -S | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; This test shows how unrolling an inner loop could break LCSSA for an outer
; loop, and there is no cheap way to recover it.
;
; In this case the inner loop, L3, is being unrolled. It only runs one
; iteration, so unrolling basically means replacing
;   br i1 true, label %exit, label %L3_header
; with
;   br label %exit
;
; However, this change messes up the loops structure: for instance, block
; L3_body no longer belongs to L2. It becomes an exit block for L2, so LCSSA
; phis for definitions in L2 should now be placed there. In particular, we need
; to insert such a definition for %y1.

; CHECK-LABEL: @foo1
define void @foo1() {
entry:
  br label %L1_header

L1_header:
  br label %L2_header

L2_header:
  %y1 = phi i64 [ undef, %L1_header ], [ %x.lcssa, %L2_latch ]
  br label %L3_header

L3_header:
  %y2 = phi i64 [ 0, %L3_latch ], [ %y1, %L2_header ]
  %x = add i64 undef, -1
  br i1 true, label %L2_latch, label %L3_body

L2_latch:
  %x.lcssa = phi i64 [ %x, %L3_header ]
  br label %L2_header

; CHECK:      L3_body:
; CHECK-NEXT:   %y1.lcssa = phi i64 [ %y1, %L3_header ]
L3_body:
  store i64 %y1, i64* undef
  br i1 false, label %L3_latch, label %L1_latch

L3_latch:
  br i1 true, label %exit, label %L3_header

L1_latch:
  %y.lcssa = phi i64 [ %y2, %L3_body ]
  br label %L1_header

exit:
  ret void
}

; Additional tests for some corner cases.
;
; CHECK-LABEL: @foo2
define void @foo2() {
entry:
  br label %L1_header

L1_header:
  br label %L2_header

L2_header:
  %a = phi i64 [ undef, %L1_header ], [ %dec_us, %L3_header ]
  br label %L3_header

L3_header:
  %b = phi i64 [ 0, %L3_latch ], [ %a, %L2_header ]
  %dec_us = add i64 undef, -1
  br i1 true, label %L2_header, label %L3_break_to_L1

; CHECK:      L3_break_to_L1:
; CHECK-NEXT:   %a.lcssa = phi i64 [ %a, %L3_header ]
L3_break_to_L1:
  br i1 false, label %L3_latch, label %L1_latch

L1_latch:
  %b_lcssa = phi i64 [ %b, %L3_break_to_L1 ]
  br label %L1_header

L3_latch:
  br i1 true, label %Exit, label %L3_header

Exit:
  ret void
}

; CHECK-LABEL: @foo3
define void @foo3() {
entry:
  br label %L1_header

L1_header:
  %a = phi i8* [ %b, %L1_latch ], [ null, %entry ]
  br i1 undef, label %L2_header, label %L1_latch

L2_header:
  br i1 undef, label %L2_latch, label %L1_latch

; CHECK:      L2_latch:
; CHECK-NEXT:   %a.lcssa = phi i8* [ %a, %L2_header ]
L2_latch:
  br i1 true, label %L2_exit, label %L2_header

L1_latch:
  %b = phi i8* [ undef, %L1_header ], [ null, %L2_header ]
  br label %L1_header

L2_exit:
  %a_lcssa1 = phi i8* [ %a, %L2_latch ]
  br label %Exit

Exit:
  %a_lcssa2 = phi i8* [ %a_lcssa1, %L2_exit ]
  ret void
}

; PR26688
; CHECK-LABEL: @foo4
define i8 @foo4() {
entry:
  br label %L1_header

L1_header:
  %x = icmp eq i32 1, 0
  br label %L2_header

L2_header:
  br label %L3_header

L3_header:
  br i1 true, label %L2_header, label %L3_exiting

L3_exiting:
  br i1 true, label %L3_body, label %L1_latch

; CHECK:      L3_body:
; CHECK-NEXT:   %x.lcssa = phi i1
L3_body:
  br i1 %x, label %L3_latch, label %L3_latch

L3_latch:
  br i1 false, label %L3_header, label %exit

L1_latch:
  br label %L1_header

exit:
  ret i8 0
}
