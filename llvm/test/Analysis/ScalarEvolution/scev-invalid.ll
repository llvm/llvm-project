; RUN: opt < %s -S -passes='loop(indvars),loop-unroll' | FileCheck %s
;
; PR15570: SEGV: SCEV back-edge info invalid after dead code removal.
;
; Indvars creates a SCEV expression for the loop's back edge taken
; count, then determines that the comparison is always true and
; removes it.
;
; When loop-unroll asks for the expression, it contains a NULL
; SCEVUnknkown (as a CallbackVH).
;
; forgetMemoizedResults should invalidate the backedge taken count expression.

; CHECK: @test
; CHECK-NOT: phi
; CHECK-NOT: icmp
; CHECK: ret void
define void @test() {
entry:
  %xor1 = xor i32 0, 1
  br label %b17

b17:
  br i1 undef, label %b22, label %b18

b18:
  %phi1 = phi i32 [ %add1, %b18 ], [ %xor1, %b17 ]
  %add1 = add nsw i32 %phi1, -1
  %cmp1 = icmp sgt i32 %add1, 0
  br i1 %cmp1, label %b18, label %b22

b22:
  ret void
}
