; RUN: llc -mtriple=riscv64 -riscv-enable-live-variables -verify-machineinstrs \
; RUN: -riscv-enable-live-variables -riscv-liveness-update-kills -stop-after=riscv-live-variables \
; RUN: -o - %s | FileCheck %s

; Test live variable analysis with loops and backward edges
; Loops create interesting liveness patterns with phi nodes and variables
; that are live across backedges

; CHECK:  test_simple_loop
; CHECK:  bb.0.entry:
; CHECK:    successors: %bb.1(0x80000000)
; CHECK:    liveins: $x10
;
; CHECK:    %4:gpr = COPY $x10
; CHECK:    %6:gpr = COPY $x0
; CHECK:    %5:gpr = COPY killed %6
;
; CHECK:  bb.1.loop:
; CHECK:    successors: %bb.1(0x7c000000), %bb.2(0x04000000)
;
; CHECK:    %0:gpr = PHI %5, %bb.0, %3, %bb.1
; CHECK:    %1:gpr = PHI %5, %bb.0, %2, %bb.1
; CHECK:    %2:gpr = ADD killed %1, %0
; CHECK:    %3:gpr = ADDI killed %0, 1
; CHECK:    BLT %3, %4, %bb.1
; CHECK:    PseudoBR %bb.2
; 
; CHECK:  bb.2.exit:
; CHECK:    $x10 = COPY killed %2
; CHECK:    PseudoRET implicit $x10

define i64 @test_simple_loop(i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %sum = phi i64 [ 0, %entry ], [ %sum.next, %loop ]
  %sum.next = add i64 %sum, %i
  %i.next = add i64 %i, 1
  %cmp = icmp slt i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret i64 %sum.next
}

; CHECK:  test_nested_loop
; CHECK:  bb.0.entry:
; CHECK:    successors: %bb.1(0x80000000)
; CHECK:    liveins: $x10, $x11
;
; CHECK:    %11:gpr = COPY $x11
; CHECK:    %10:gpr = COPY $x10
; CHECK:    %13:gpr = COPY $x0
; CHECK:    %12:gpr = COPY killed %13
;
; CHECK:  bb.1.outer.loop:
; CHECK:    successors: %bb.2(0x80000000)
;
; CHECK:    %0:gpr = PHI %12, %bb.0, %9, %bb.3
; CHECK:    %1:gpr = PHI %12, %bb.0, %8, %bb.3
; CHECK:    %15:gpr = COPY $x0
; CHECK:    %14:gpr = COPY killed %15
;
; CHECK:  bb.2.inner.loop:
; CHECK:    successors: %bb.2(0x7c000000), %bb.3(0x04000000)
;
; CHECK:    %2:gpr = PHI %14, %bb.1, %7, %bb.2
; CHECK:    %3:gpr = PHI %14, %bb.1, %6, %bb.2
; CHECK:    %4:gpr = PHI %14, %bb.1, %5, %bb.2
; CHECK:    %5:gpr = ADD %4, %2
; CHECK:    %6:gpr = ADDI killed %3, 1
; CHECK:    %7:gpr = ADD killed %2, %0
; CHECK:    BLT %6, %11, %bb.2
; CHECK:    PseudoBR %bb.3
;
; CHECK:  bb.3.outer.latch:
; CHECK:    successors: %bb.1(0x7c000000), %bb.4(0x04000000)
;
; CHECK:    %8:gpr = ADD killed %1, %5
; CHECK:    %9:gpr = ADDI killed %0, 1
; CHECK:    BLT %9, %10, %bb.1
; CHECK:    PseudoBR %bb.4
;
; CHECK:  bb.4.exit:
; CHECK:    $x10 = COPY killed %8
; CHECK:    PseudoRET implicit $x10

define i64 @test_nested_loop(i64 %n, i64 %m) {
entry:
  br label %outer.loop

outer.loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %outer.sum = phi i64 [ 0, %entry ], [ %new.outer.sum, %outer.latch ]
  br label %inner.loop

inner.loop:
  %j = phi i64 [ 0, %outer.loop ], [ %j.next, %inner.loop ]
  %inner.sum = phi i64 [ 0, %outer.loop ], [ %inner.sum.next, %inner.loop ]
  %prod = mul i64 %i, %j
  %inner.sum.next = add i64 %inner.sum, %prod
  %j.next = add i64 %j, 1
  %inner.cmp = icmp slt i64 %j.next, %m
  br i1 %inner.cmp, label %inner.loop, label %outer.latch

outer.latch:
  %new.outer.sum = add i64 %outer.sum, %inner.sum.next
  %i.next = add i64 %i, 1
  %outer.cmp = icmp slt i64 %i.next, %n
  br i1 %outer.cmp, label %outer.loop, label %exit

exit:
  ret i64 %new.outer.sum
}

; CHECK:  test_loop_with_invariant
; CHECK:  bb.0.entry:
; CHECK:    successors: %bb.1(0x80000000)
; CHECK:    liveins: $x10, $x11
;
; CHECK:    %8:gpr = COPY $x11
; CHECK:    %7:gpr = COPY $x10
; CHECK:    %0:gpr = SLLI killed %8, 1
; CHECK:    %10:gpr = COPY $x0
; CHECK:    %9:gpr = COPY killed %10
;
; CHECK:  bb.1.loop:
; CHECK:    successors: %bb.1(0x7c000000), %bb.2(0x04000000)
;
; CHECK:    %1:gpr = PHI %9, %bb.0, %6, %bb.1
; CHECK:    %2:gpr = PHI %9, %bb.0, %5, %bb.1
; CHECK:    %3:gpr = PHI %9, %bb.0, %4, %bb.1
; CHECK:    %4:gpr = ADD killed %3, %1
; CHECK:    %5:gpr = ADDI killed %2, 1
; CHECK:    %6:gpr = ADD killed %1, %0
; CHECK:    BLT %5, %7, %bb.1
; CHECK:    PseudoBR %bb.2
;
; CHECK:  bb.2.exit:
; CHECK:    $x10 = COPY killed %4
; CHECK:    PseudoRET implicit $x10

define i64 @test_loop_with_invariant(i64 %n, i64 %k) {
entry:
  %double_k = mul i64 %k, 2
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %sum = phi i64 [ 0, %entry ], [ %sum.next, %loop ]
  ; double_k is loop-invariant and should be live throughout the loop
  %scaled = mul i64 %i, %double_k
  %sum.next = add i64 %sum, %scaled
  %i.next = add i64 %i, 1
  %cmp = icmp slt i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret i64 %sum.next
}
