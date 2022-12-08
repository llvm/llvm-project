; RUN: not opt -S %s -passes=verify 2>&1 | FileCheck %s

; CHECK: Number of label constraints does not match number of callbr dests
; CHECK-NEXT: #too_few_label_constraints
define void @too_few_label_constraints() {
  callbr void asm sideeffect "#too_few_label_constraints", "!i"()
  to label %1 [label %2, label %3]
1:
  ret void
2:
  ret void
3:
  ret void
}

; CHECK-NOT: Number of label constraints does not match number of callbr dests
define void @correct_label_constraints() {
  callbr void asm sideeffect "${0:l} ${1:l}", "!i,!i"()
  to label %1 [label %2, label %3]
1:
  ret void
2:
  ret void
3:
  ret void
}

; CHECK: Number of label constraints does not match number of callbr dests
; CHECK-NEXT: #too_many_label_constraints
define void @too_many_label_constraints() {
  callbr void asm sideeffect "#too_many_label_constraints", "!i,!i,!i"()
  to label %1 [label %2, label %3]
1:
  ret void
2:
  ret void
3:
  ret void
}

; CHECK: Label constraints can only be used with callbr
; CHECK-NEXT: #label_constraint_without_callbr
define void @label_constraint_without_callbr() {
  call void asm sideeffect "#label_constraint_without_callbr", "!i"()
  ret void
}

; CHECK: Number of label constraints does not match number of callbr dests
; CHECK-NEXT: #callbr_without_label_constraint
define void @callbr_without_label_constraint() {
  callbr void asm sideeffect "#callbr_without_label_constraint", ""()
  to label %1 [label %2]
1:
  ret void
2:
  ret void
}

;; Ensure you cannot use the return value of a callbr in indirect targets.
; CHECK: Instruction does not dominate all uses!
; CHECK-NEXT: #test4
define i32 @test4(i1 %var) {
entry:
  %ret = callbr i32 asm sideeffect "#test4", "=r,!i"() to label %normal [label %abnormal]

normal:
  ret i32 0

abnormal:
  ret i32 %ret
}
