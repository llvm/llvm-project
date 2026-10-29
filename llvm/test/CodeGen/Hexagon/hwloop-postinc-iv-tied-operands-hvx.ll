; RUN: llc -mtriple=hexagon -mattr=+hvxv68,+hvx-length128b -O2 < %s \
; RUN:   | FileCheck %s
;
; HVX counterparts of hwloop-postinc-iv-tied-operands.ll.  The vector
; post-increment forms have the same two operand shapes as the scalar ones,
; so the tied def index likewise differs by opcode:
;
;   V6_vS32b_pi    (outs $Rx32),         (ins $Rx32in, $Ii, $Vs32)  -> tied def 0
;   V6_vL32b_pi    (outs $Vd32, $Rx32),  (ins $Rx32in, $Ii)         -> tied def 1
;
; Both loops have a genuine "base + constant" induction variable (the PHI is
; fed by the incremented address), so a hardware loop must be formed.

; --- V6_vS32b_pi: single def, tied def index 0 -----------------------------
; CHECK-LABEL: vstore_pi_tied0:
; CHECK: loop0
define void @vstore_pi_tied0(ptr %begin, ptr %end, <32 x i32> %v) {
entry:
  %empty = icmp eq ptr %begin, %end
  br i1 %empty, label %exit, label %loop

loop:
  %ptr = phi ptr [ %begin, %entry ], [ %ptr.next, %loop ]
  %old = load <32 x i32>, ptr %ptr, align 128
  %sum = add <32 x i32> %old, %v
  store <32 x i32> %sum, ptr %ptr, align 128
  %ptr.next = getelementptr inbounds <32 x i32>, ptr %ptr, i32 1
  %done = icmp eq ptr %ptr.next, %end
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; --- V6_vL32b_pi: two defs, tied def index 1 ------------------------------
; CHECK-LABEL: vload_pi_tied1:
; CHECK: loop0
define <32 x i32> @vload_pi_tied1(ptr %begin, ptr %end) {
entry:
  %empty = icmp eq ptr %begin, %end
  br i1 %empty, label %exit, label %loop

loop:
  %ptr = phi ptr [ %begin, %entry ], [ %ptr.next, %loop ]
  %acc = phi <32 x i32> [ zeroinitializer, %entry ], [ %acc.next, %loop ]
  %val = load <32 x i32>, ptr %ptr, align 128
  %acc.next = add <32 x i32> %acc, %val
  %ptr.next = getelementptr inbounds <32 x i32>, ptr %ptr, i32 1
  %done = icmp eq ptr %ptr.next, %end
  br i1 %done, label %exit, label %loop

exit:
  %result = phi <32 x i32> [ zeroinitializer, %entry ], [ %acc.next, %loop ]
  ret <32 x i32> %result
}
