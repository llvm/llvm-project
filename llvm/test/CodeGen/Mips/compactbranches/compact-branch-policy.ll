; Check that -mips-compact-branches={never,optimal,always} is accepted and honoured.
; RUN: llc -march=mips -mcpu=mips32r6 -mips-compact-branches=never < %s | FileCheck %s -check-prefix=NEVER
; RUN: llc -march=mips -mcpu=mips32r6 -mips-compact-branches=optimal < %s | FileCheck %s -check-prefix=OPTIMAL
; RUN: llc -march=mips -mcpu=mips32r6 -mips-compact-branches=always < %s | FileCheck %s -check-prefix=ALWAYS

define i32 @l(i32 signext %a, i32 signext %b) {
entry:
  %add = add nsw i32 %b, %a
  %cmp = icmp slt i32 %add, 100
; NEVER: beq
; OPTIMAL: beq
; ALWAYS: beqzc
; This nop is required for correct as having (j|b)al as the instruction
; immediately following beqzc would cause a forbidden slot hazard.
; ALWAYS: nop
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call = tail call i32 @k()
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %call.pn = phi i32 [ %call, %if.then ], [ -1, %entry ]
  %c.0 = add nsw i32 %call.pn, %add
  ret i32 %c.0
}

declare i32 @k() #1
