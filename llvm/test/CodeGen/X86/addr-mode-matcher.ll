; RUN: llc < %s | FileCheck %s

; This testcase used to hit an assert during ISel.  For details, see the big
; comment inside the function.

; CHECK-LABEL: foo:
; The AND should be turned into a subreg access.
; CHECK-NOT: and
; The shift (leal) should be folded into the scale of the address in the load.
; CHECK-NOT: leal
; CHECK: movl {{.*}},4),

target datalayout = "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.6.0"

define void @foo(i32 %a) {
bb:
  br label %bb1692

bb1692:
  %tmp1694 = phi i32 [ 0, %bb ], [ %tmp1745, %bb1692 ]
  %xor = xor i32 0, %tmp1694

; %load1 = (load (and (shl %xor, 2), 1020))
  %tmp1701 = shl i32 %xor, 2
  %tmp1702 = and i32 %tmp1701, 1020
  %tmp1703 = getelementptr inbounds [1028 x i8], ptr null, i32 0, i32 %tmp1702
  %load1 = load i32, ptr %tmp1703, align 4

; %load2 = (load (shl (and %xor, 255), 2))
  %tmp1698 = and i32 %xor, 255
  %tmp1706 = shl i32 %tmp1698, 2
  %tmp1707 = getelementptr inbounds [1028 x i8], ptr null, i32 0, i32 %tmp1706
  %load2 = load i32, ptr %tmp1707, align 4

  %tmp1710 = or i32 %load2, %a

; While matching xor we address-match %load1.  The and-of-shift reassocication
; in address matching transform this into into a shift-of-and and the resuting
; node becomes identical to %load2.  CSE replaces %load1 which leaves its
; references in MatchScope and RecordedNodes stale.
  %tmp1711 = xor i32 %load1, %tmp1710

  %tmp1744 = getelementptr inbounds [256 x i32], ptr null, i32 0, i32 %tmp1711
  store i32 0, ptr %tmp1744, align 4
  %tmp1745 = add i32 %tmp1694, 1
  indirectbr ptr undef, [label %bb1756, label %bb1692]

bb1756:
  br label %bb2705

bb2705:
  indirectbr ptr undef, [label %bb5721, label %bb5736]

bb5721:
  br label %bb2705

bb5736:
  ret void
}
