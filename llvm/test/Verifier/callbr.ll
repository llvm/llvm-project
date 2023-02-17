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

;; Ensure you can use the return value of a callbr in indirect targets.
;; No issue!
define i32 @test4(i1 %var) {
entry:
  %ret = callbr i32 asm sideeffect "#test4", "=r,!i"() to label %normal [label %abnormal]

normal:
  ret i32 0

abnormal:
  ret i32 %ret
}

;; Tests of the callbr.landingpad intrinsic function.
declare i32 @llvm.callbr.landingpad.i64(i64)
define void @callbrpad_bad_type() {
entry:
; CHECK: Intrinsic has incorrect argument type!
; CHECK-NEXT: ptr @llvm.callbr.landingpad.i64
  %foo = call i32 @llvm.callbr.landingpad.i64(i64 42)
  ret void
}

declare i32 @llvm.callbr.landingpad.i32(i32)
define i32 @callbrpad_multi_preds() {
entry:
  %foo = callbr i32 asm "", "=r,!i"() to label %direct [label %indirect]
direct:
  br label %indirect
indirect:
; CHECK-NEXT: Intrinsic in block must have 1 unique predecessor
; CHECK-NEXT: %out = call i32 @llvm.callbr.landingpad.i32(i32 %foo)
  %out = call i32 @llvm.callbr.landingpad.i32(i32 %foo)
  ret i32 %out
}

define void @callbrpad_wrong_callbr() {
entry:
  %foo = callbr i32 asm "", "=r,!i"() to label %direct [label %indirect]
direct:
; CHECK-NEXT: Intrinsic's corresponding callbr must have intrinsic's parent basic block in indirect destination list
; CHECK-NEXT: %x = call i32 @llvm.callbr.landingpad.i32(i32 %foo)
  %x = call i32 @llvm.callbr.landingpad.i32(i32 %foo)
  ret void
indirect:
  ret void
}

declare i32 @foo(i32)
define i32 @test_callbr_landingpad_not_first_inst() {
entry:
  %0 = callbr i32 asm "", "=r,!i"()
          to label %asm.fallthrough [label %landingpad]

asm.fallthrough:
  ret i32 42

landingpad:
  %foo = call i32 @foo(i32 42)
; CHECK-NEXT: No other instructions may proceed intrinsic
; CHECK-NEXT: %out = call i32 @llvm.callbr.landingpad.i32(i32 %0)
  %out = call i32 @llvm.callbr.landingpad.i32(i32 %0)
  ret i32 %out
}
