; RUN: llc -march=hexagon -enable-pipeliner -verify-machineinstrs < %s
; REQUIRES: asserts

; This test fails in the machine verifier because the verifier thinks the
; return register is undefined, and because there is a basic block that
; ends with an unconditional branch that is not marked as a barrier.
;
; Enabling SWP exposes these bugs because the live variable analysis is
; performed earlier than the process implicit def pass.  This ordering
; causes the JMPR machine instruction to contain two R0 operands, one
; with an undef and one with a kill flag.

@g0 = common global i32 0, align 4

; Function Attrs: nounwind
define i32 @f0(i32 %a0) #0 {
b0:
  %v0 = icmp eq i32 %a0, 0
  br i1 %v0, label %b2, label %b1

b1:                                               ; preds = %b0
  %v1 = tail call i32 @f1(i32 %a0) #0
  br label %b3

b2:                                               ; preds = %b0
  store i32 0, ptr @g0, align 4
  br label %b3

b3:                                               ; preds = %b2, %b1
  ret i32 undef
}

declare i32 @f1(...)

attributes #0 = { nounwind }
