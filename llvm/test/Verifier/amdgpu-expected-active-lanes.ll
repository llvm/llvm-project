; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK-DAG: !amdgpu.expected.active.lanes must have exactly one operand
define i32 @no_operands(ptr %p, i32 %v) {
  %old = atomicrmw add ptr %p, i32 %v seq_cst, !amdgpu.expected.active.lanes !0
  ret i32 %old
}

; CHECK-DAG: !amdgpu.expected.active.lanes must have exactly one operand
define i32 @too_many_operands(ptr %p, i32 %v) {
  %old = atomicrmw add ptr %p, i32 %v seq_cst, !amdgpu.expected.active.lanes !1
  ret i32 %old
}

; CHECK-DAG: !amdgpu.expected.active.lanes operand must be an i32 constant
define i32 @operand_not_a_constant(ptr %p, i32 %v) {
  %old = atomicrmw add ptr %p, i32 %v seq_cst, !amdgpu.expected.active.lanes !2
  ret i32 %old
}

; CHECK-DAG: !amdgpu.expected.active.lanes operand must be an i32 constant
define i32 @operand_wrong_width(ptr %p, i32 %v) {
  %old = atomicrmw add ptr %p, i32 %v seq_cst, !amdgpu.expected.active.lanes !3
  ret i32 %old
}

!0 = !{}
!1 = !{i32 4, i32 5}
!2 = !{!"not a constant"}
!3 = !{i64 4}
