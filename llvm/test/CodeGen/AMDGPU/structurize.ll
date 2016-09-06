; RUN: llc < %s -march=r600 -mcpu=redwood -r600-ir-structurize=0 | FileCheck %s
; Test case for a crash in the AMDILCFGStructurizer from a CFG like this:
;
;                            entry
;                           /     \
;               diamond_head       branch_from
;                 /      \           |
;    diamond_false        diamond_true
;                 \      /
;                   done
;
; When the diamond_true branch had more than 100 instructions.
;
;

; CHECK-LABEL: {{^}}branch_into_diamond:
; === entry block:
; CHECK: ALU_PUSH_BEFORE
; === Branch instruction (IF):
; CHECK: JUMP
  ; === branch_from block
  ; CHECK: ALU
  ; === Duplicated diamond_true block (There can be more than one ALU clause):
  ; === XXX: We should be able to optimize this so the basic block is not
  ; === duplicated.  See comments in
  ; === AMDGPUCFGStructurizer::improveSimpleJumpintoIf()
  ; CHECK: ALU
; === Branch instruction (ELSE):
; CHECK: ELSE
  ; === diamond_head block:
  ; CHECK: ALU_PUSH_BEFORE
  ; === Branch instruction (IF):
  ; CHECK: JUMP
    ; === diamond_true block (There can be more than one ALU clause):
    ; ALU
  ; === Branch instruction (ELSE):
  ; CHECK: ELSE
    ; === diamond_false block plus implicit ENDIF
    ; CHECK: ALU_POP_AFTER
; === Branch instruction (ENDIF):
; CHECK: POP
; === done block:
; CHECK: ALU
; CHECK: MEM_RAT_CACHELESS
; CHECK: CF_END


define void @branch_into_diamond(i32 addrspace(1)* %out, i32 %a, i32 %b, i32 %c) {
entry:
%0 = icmp ne i32 %a, 0
  br i1 %0, label %diamond_head, label %branch_from

diamond_head:
  %1 = icmp ne i32 %a, 1
  br i1 %1, label %diamond_true, label %diamond_false

branch_from:
  %2 = add i32 %a, 1
  br label %diamond_true

diamond_false:
  %3 = add i32 %a, 2
  br label %done

diamond_true:
  %4 = phi i32 [%2, %branch_from], [%a, %diamond_head]
  ; This block needs to be > 100 ISA instructions to hit the bug,
  ; so we'll use udiv instructions.
  %div0 = udiv i32 %a, %b
  %div1 = udiv i32 %div0, %4
  %div2 = udiv i32 %div1, 11
  %div3 = udiv i32 %div2, %a
  %div4 = udiv i32 %div3, %b
  %div5 = udiv i32 %div4, %c
  %div6 = udiv i32 %div5, %div0
  %div7 = udiv i32 %div6, %div1
  br label %done

done:
  %5 = phi i32 [%3, %diamond_false], [%div7, %diamond_true]
  store i32 %5, i32 addrspace(1)* %out
  ret void
}
