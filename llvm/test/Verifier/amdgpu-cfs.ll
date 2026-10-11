; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK-DAG: !amdgpu.cfs must have exactly one operand
define i32 @no_operands(ptr %p) {
  %v = load i32, ptr %p, !amdgpu.cfs !0
  ret i32 %v
}

; CHECK-DAG: !amdgpu.cfs must have exactly one operand
define i32 @too_many_operands(ptr %p) {
  %v = load i32, ptr %p, !amdgpu.cfs !1
  ret i32 %v
}

; CHECK-DAG: !amdgpu.cfs operand must be an i32 constant
define i32 @operand_not_a_constant(ptr %p) {
  %v = load i32, ptr %p, !amdgpu.cfs !2
  ret i32 %v
}

; CHECK-DAG: !amdgpu.cfs operand must be an i32 constant
define i32 @operand_wrong_width(ptr %p) {
  %v = load i32, ptr %p, !amdgpu.cfs !3
  ret i32 %v
}

; CHECK-DAG: !amdgpu.cfs operand must be in the range [0, 3]
define i32 @operand_out_of_range(ptr %p) {
  %v = load i32, ptr %p, !amdgpu.cfs !4
  ret i32 %v
}

; CHECK-DAG: !amdgpu.cfs operand must be in the range [0, 3]
define i32 @operand_negative(ptr %p) {
  %v = load i32, ptr %p, !amdgpu.cfs !5
  ret i32 %v
}

; CHECK-DAG: !amdgpu.cfs operand must be an i32 constant
define i32 @operand_null(ptr %p) {
  %v = load i32, ptr %p, !amdgpu.cfs !6
  ret i32 %v
}
; CHECK-DAG: !amdgpu.cfs operand must be in the range [0, 3]
define i32 @operand_just_out_of_range(ptr %p) {
  %v = load i32, ptr %p, !amdgpu.cfs !7
  ret i32 %v
}

!0 = !{}
!1 = !{i32 1, i32 2}
!2 = !{!"not a constant"}
!3 = !{i64 1}
!4 = !{i32 7}
!5 = !{i32 -1}
!6 = !{null}
!7 = !{i32 4}
