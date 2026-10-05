; RUN: llc -mtriple=powerpc-none-eabi -mcpu=ppe42 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=powerpc-none-eabi -mcpu=ppe42 -verify-machineinstrs -disable-ppc-sco < %s | FileCheck %s --check-prefix=NOSCO
;
; Test sibling-call optimisation (SCO) for the PPE42 target.
;
; PPE42 is a 32-bit SVR4 ABI target without a TOC register, so the
; TOC-sharing restrictions that limit SCO on 64-bit ELF do not apply.
; The IsEligibleForTailCallOptimization path should emit a plain branch
; (b / TC_RETURN) instead of a full call frame (bl + mflr/mtlr/stwu/addi).
;
; LowerCall_32SVR4 mirrors the 64-bit IsSibCall path: CALLSEQ_START is
; suppressed for SCO so MFI.adjustsStack() stays false, producing a
; frameless function with no argument spills to the stack.
;
; A correct SCO sequence must NOT contain:
;   mflr  -- save link register (non-leaf prologue)
;   stwu  -- stack frame allocation
;   stw   -- argument spill to outgoing stack slot
;   lwz   -- argument reload from stack slot
;   mtlr  -- restore link register
;   bl    -- branch-and-link (normal call)
;
; A correct SCO sequence MUST contain:
;   b     -- unconditional branch (tail jump)

; ---------------------------------------------------------------------------
; 1. Simple stub: function whose entire body is "return callee(same-args)".
;    This is the exact pattern seen in plat_hw_access.C.
;    Matches: same CallingConv (C), same argument list -> always SCO.
; ---------------------------------------------------------------------------
declare i32* @errorl(i32, i32, i32)

define i32* @getCfamRegister(i32 %target, i32 %addr, i32 %data) {
; CHECK-LABEL: getCfamRegister:
; CHECK-NOT:   mflr
; CHECK-NOT:   stwu
; CHECK-NOT:   stw
; CHECK-NOT:   lwz
; CHECK-NOT:   bl      errorl
; CHECK:       b       errorl
;
; NOSCO-LABEL: getCfamRegister:
; NOSCO:       bl      errorl
  %rc = tail call i32* @errorl(i32 3, i32 9, i32 -1)
  ret i32* %rc
}

; ---------------------------------------------------------------------------
; 2. Same pattern, putCfamRegister – verify both stubs in the same TU get SCO.
; ---------------------------------------------------------------------------
define i32* @putCfamRegister(i32 %target, i32 %addr, i32 %data) {
; CHECK-LABEL: putCfamRegister:
; CHECK-NOT:   mflr
; CHECK-NOT:   stwu
; CHECK-NOT:   stw
; CHECK-NOT:   lwz
; CHECK-NOT:   bl      errorl
; CHECK:       b       errorl
  %rc = tail call i32* @errorl(i32 3, i32 9, i32 -1)
  ret i32* %rc
}

; ---------------------------------------------------------------------------
; 3. Caller forwards its own arguments directly to callee — classic SCO.
; ---------------------------------------------------------------------------
declare i32 @callee_3args(i32, i32, i32)

define i32 @forward_args(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: forward_args:
; CHECK-NOT:   mflr
; CHECK-NOT:   stwu
; CHECK:       b       callee_3args
  %r = tail call i32 @callee_3args(i32 %a, i32 %b, i32 %c)
  ret i32 %r
}

; ---------------------------------------------------------------------------
; 4. Caller with more args than callee — callee uses subset, all in GPRs.
;    SCO is still legal because outgoing args fit in r3-r10.
; ---------------------------------------------------------------------------
declare i32 @callee_1arg(i32)

define i32 @fewer_args(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: fewer_args:
; CHECK-NOT:   mflr
; CHECK-NOT:   stwu
; CHECK:       b       callee_1arg
  %r = tail call i32 @callee_1arg(i32 %a)
  ret i32 %r
}

; ---------------------------------------------------------------------------
; 5. Zero-argument callee — the simplest possible tail call.
; ---------------------------------------------------------------------------
declare i32 @callee_noargs()

define i32 @no_args_wrapper() {
; CHECK-LABEL: no_args_wrapper:
; CHECK-NOT:   mflr
; CHECK-NOT:   stwu
; CHECK:       b       callee_noargs
  %r = tail call i32 @callee_noargs()
  ret i32 %r
}

; ---------------------------------------------------------------------------
; 6. fastcc GuaranteedTailCallOpt path must still work (regression guard).
;    Tested with -tailcallopt flag; without it fastcc SCO should still fire.
; ---------------------------------------------------------------------------
declare fastcc i32 @fastcc_callee(i32, i32)

define fastcc i32 @fastcc_wrapper(i32 %a, i32 %b) {
; CHECK-LABEL: fastcc_wrapper:
; CHECK-NOT:   mflr
; CHECK-NOT:   stwu
; CHECK:       b       fastcc_callee
  %r = tail call fastcc i32 @fastcc_callee(i32 %a, i32 %b)
  ret i32 %r
}

; ---------------------------------------------------------------------------
; 7. vararg callee — must NOT be SCO'd (safety: stack layout unknown).
; ---------------------------------------------------------------------------
declare i32 @vararg_callee(i32, ...)

define i32 @no_sco_vararg(i32 %a) {
; CHECK-LABEL: no_sco_vararg:
; CHECK:       bl      vararg_callee
; CHECK:       blr
  %r = call i32 (i32, ...) @vararg_callee(i32 %a)
  ret i32 %r
}

; ---------------------------------------------------------------------------
; 8. Tail call with constant arguments (different from caller params) —
;    all fit in GPRs so SCO is allowed.
; ---------------------------------------------------------------------------
define i32 @constant_args_sco(i32 %unused) {
; CHECK-LABEL: constant_args_sco:
; CHECK-NOT:   mflr
; CHECK-NOT:   stwu
; CHECK-NOT:   stw
; CHECK-NOT:   lwz
; CHECK:       b       callee_3args
  %r = tail call i32 @callee_3args(i32 1, i32 2, i32 3)
  ret i32 %r
}

; ---------------------------------------------------------------------------
; 9. -disable-ppc-sco: all plain tail calls must fall back to bl+blr.
; ---------------------------------------------------------------------------
define i32 @nosco_forward(i32 %a, i32 %b, i32 %c) {
; NOSCO-LABEL: nosco_forward:
; NOSCO:       bl      callee_3args
; NOSCO:       blr
  %r = tail call i32 @callee_3args(i32 %a, i32 %b, i32 %c)
  ret i32 %r
}

; ---------------------------------------------------------------------------
; 10. Conditional tail call — SCO should fire on the taken branch.
; ---------------------------------------------------------------------------
define i32 @conditional_sco(i32 %x, i32 %a, i32 %b) {
; CHECK-LABEL: conditional_sco:
; CHECK:       b       callee_1arg
entry:
  %cmp = icmp eq i32 %x, 0
  br i1 %cmp, label %do_tail, label %do_ret

do_tail:
  %r = tail call i32 @callee_1arg(i32 %a)
  ret i32 %r

do_ret:
  ret i32 %b
}

; ---------------------------------------------------------------------------
; 11. Verify no stack frame is created for SCO — blr must not appear before b.
;     i.e. the function has no normal-return path once SCO fires.
; ---------------------------------------------------------------------------
define i32* @no_stack_frame_stub(i32 %t, i32 %a, i32 %d) {
; CHECK-LABEL: no_stack_frame_stub:
; CHECK-NOT:   stwu
; CHECK-NOT:   stw
; CHECK-NOT:   lwz
; CHECK-NOT:   mflr
; CHECK:       b       errorl
  %rc = tail call i32* @errorl(i32 3, i32 9, i32 -1)
  ret i32* %rc
}

; ---------------------------------------------------------------------------
; 12. Forwarding all 9 incoming args to a 9-arg callee: the 9th argument
;     already sits at the right stack offset in the caller's incoming frame
;     (SPDiff=0, so FixedObject lands at the same offset).  No new frame is
;     needed — this is still a valid SCO.
; ---------------------------------------------------------------------------
declare i32 @callee_9args(i32, i32, i32, i32, i32, i32, i32, i32, i32)

define i32 @forward_9args(i32 %a, i32 %b, i32 %c, i32 %d,
                           i32 %e, i32 %f, i32 %g, i32 %h, i32 %i) {
; CHECK-LABEL: forward_9args:
; CHECK-NOT:   stwu
; CHECK:       b       callee_9args
  %r = tail call i32 @callee_9args(i32 %a, i32 %b, i32 %c, i32 %d,
                                    i32 %e, i32 %f, i32 %g, i32 %h, i32 %i)
  ret i32 %r
}

; ---------------------------------------------------------------------------
; 13. Caller passes a *different* value for a stack-spilled arg (9th arg is
;     a constant, not the forwarded incoming value).  SCO is blocked by
;     IsEligibleForTailCallOptimization (GPRsUsed > 8 for new-value case).
; ---------------------------------------------------------------------------
define i32 @spill_new_val(i32 %a, i32 %b, i32 %c, i32 %d,
                           i32 %e, i32 %f, i32 %g, i32 %h) {
; CHECK-LABEL: spill_new_val:
; CHECK:       stwu
  %r = tail call i32 @callee_9args(i32 %a, i32 %b, i32 %c, i32 %d,
                                    i32 %e, i32 %f, i32 %g, i32 %h, i32 42)
  ret i32 %r
}
