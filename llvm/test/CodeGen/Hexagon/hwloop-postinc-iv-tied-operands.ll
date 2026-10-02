; RUN: llc -mtriple=hexagon -O2 < %s | FileCheck %s
;
; HexagonHardwareLoops recognizes a post-increment memory instruction as the
; induction-variable bump.  Such instructions tie their base-address use to
; the def that holds the incremented address, but the index of that tied def
; depends on the opcode shape:
;
;   S2_storeri_pi  (outs $Rx32),         (ins $Rx32in, $Ii, $Rt32)  -> tied def 0
;   L2_loadri_pi   (outs $Rd32, $Rx32),  (ins $Rx32in, $Ii)         -> tied def 1
;
; The pass must locate the incremented address through the base operand's tie
; rather than a fixed operand index, and must require that the register fed
; back into the loop PHI is that incremented address.  A post-increment load
; also defines the *loaded value*, which bears no relation to the base; see
; hwloop-postinc-iv-ptrchase.ll for the miscompile that results from
; confusing the two.  The HVX vector forms are covered by
; hwloop-postinc-iv-tied-operands-hvx.ll.
;
; These are the positive cases: each loop has a genuine "base + constant"
; induction variable, so a hardware loop must still be formed.

; --- S2_storeri_pi: single def, tied def index 0 ---------------------------
; CHECK-LABEL: store_pi_tied0:
; CHECK: loop0
define void @store_pi_tied0(ptr %begin, ptr %end, i32 %v) {
entry:
  %empty = icmp eq ptr %begin, %end
  br i1 %empty, label %exit, label %loop

loop:
  %ptr = phi ptr [ %begin, %entry ], [ %ptr.next, %loop ]
  store i32 %v, ptr %ptr, align 4
  %ptr.next = getelementptr inbounds i32, ptr %ptr, i32 1
  %done = icmp eq ptr %ptr.next, %end
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; --- L2_loadri_pi: two defs, tied def index 1 -----------------------------
; The PHI is fed by the incremented address, not the loaded value, so this
; is a real induction variable.
; CHECK-LABEL: load_pi_tied1:
; CHECK: loop0
define i32 @load_pi_tied1(ptr %begin, ptr %end) {
entry:
  %empty = icmp eq ptr %begin, %end
  br i1 %empty, label %exit, label %loop

loop:
  %ptr = phi ptr [ %begin, %entry ], [ %ptr.next, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %loop ]
  %val = load i32, ptr %ptr, align 4
  %acc.next = add i32 %acc, %val
  %ptr.next = getelementptr inbounds i32, ptr %ptr, i32 1
  %done = icmp eq ptr %ptr.next, %end
  br i1 %done, label %exit, label %loop

exit:
  %result = phi i32 [ 0, %entry ], [ %acc.next, %loop ]
  ret i32 %result
}
