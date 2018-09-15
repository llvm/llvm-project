; RUN: llc -mtriple armv7-arm-linux-gnueabihf -O2 -mcpu=cortex-a7 < %s | FileCheck %s

; Check support for returning a float in GPR with soft float ABI
define arm_aapcscc float @zerobits_float_soft() #0 {
; CHECK-LABEL: zerobits_float_soft
; CHECK: mov r0, #0
  %1 = tail call float asm "mov ${0}, #0", "=&r"()
  ret float %1
}

; Check support for returning a double in GPR with soft float ABI
define arm_aapcscc double @zerobits_double_soft() #0 {
; CHECK-LABEL: zerobits_double_soft
; CHECK: mov r0, #0
; CHECK-NEXT: mov r1, #0
  %1 = tail call double asm "mov ${0:Q}, #0\0Amov ${0:R}, #0", "=&r"()
  ret double %1
}

; Check support for returning a float in GPR with matching float input with
; soft float ABI
define arm_aapcscc float @flt_gpr_matching_in_op_soft(float %f) #0 {
; CHECK-LABEL: flt_gpr_matching_in_op_soft
; CHECK: mov r0, r0
  %1 = call float asm "mov $0, $1", "=&r,0"(float %f)
  ret float %1
}

; Check support for returning a double in GPR with matching double input with
; soft float ABI
define arm_aapcscc double @dbl_gpr_matching_in_op_soft(double %d) #0 {
; CHECK-LABEL: dbl_gpr_matching_in_op_soft
; CHECK: mov r1, r0
  %1 = call double asm "mov ${0:R}, ${1:Q}", "=&r,0"(double %d)
  ret double %1
}

; Check support for returning a float in specific GPR with matching float input
; with soft float ABI
define arm_aapcscc float @flt_gpr_matching_spec_reg_in_op_soft(float %f) #0 {
; CHECK-LABEL: flt_gpr_matching_spec_reg_in_op_soft
; CHECK: mov r3, r3
  %1 = call float asm "mov $0, $1", "=&{r3},0"(float %f)
  ret float %1
}

; Check support for returning a double in specific GPR with matching double
; input with soft float ABI
define arm_aapcscc double @dbl_gpr_matching_spec_reg_in_op_soft(double %d) #0 {
; CHECK-LABEL: dbl_gpr_matching_spec_reg_in_op_soft
; CHECK: mov r3, r2
  %1 = call double asm "mov ${0:R}, ${1:Q}", "=&{r2},0"(double %d)
  ret double %1
}

attributes #0 = { nounwind "target-features"="+d16,+vfp2,+vfp3,-fp-only-sp" "use-soft-float"="true" }


; Check support for returning a float in GPR with hard float ABI
define float @zerobits_float_hard() #1 {
; CHECK-LABEL: zerobits_float_hard
; CHECK: mov r0, #0
; CHECK: vmov s0, r0
  %1 = tail call float asm "mov ${0}, #0", "=&r"()
  ret float %1
}

; Check support for returning a double in GPR with hard float ABI
define double @zerobits_double_hard() #1 {
; CHECK-LABEL: zerobits_double_hard
; CHECK: mov r0, #0
; CHECK-NEXT: mov r1, #0
; CHECK: vmov d0, r0, r1
  %1 = tail call double asm "mov ${0:Q}, #0\0Amov ${0:R}, #0", "=&r"()
  ret double %1
}

; Check support for returning a float in GPR with matching float input with
; hard float ABI
define float @flt_gpr_matching_in_op_hard(float %f) #1 {
; CHECK-LABEL: flt_gpr_matching_in_op_hard
; CHECK: vmov r0, s0
; CHECK: mov r0, r0
; CHECK: vmov s0, r0
  %1 = call float asm "mov $0, $1", "=&r,0"(float %f)
  ret float %1
}

; Check support for returning a double in GPR with matching double input with
; hard float ABI
define double @dbl_gpr_matching_in_op_hard(double %d) #1 {
; CHECK-LABEL: dbl_gpr_matching_in_op_hard
; CHECK: vmov r0, r1, d0
; CHECK: mov r1, r0
; CHECK: vmov d0, r0, r1
  %1 = call double asm "mov ${0:R}, ${1:Q}", "=&r,0"(double %d)
  ret double %1
}

; Check support for returning a float in specific GPR with matching float
; input with hard float ABI
define float @flt_gpr_matching_spec_reg_in_op_hard(float %f) #1 {
; CHECK-LABEL: flt_gpr_matching_spec_reg_in_op_hard
; CHECK: vmov r3, s0
; CHECK: mov r3, r3
; CHECK: vmov s0, r3
  %1 = call float asm "mov $0, $1", "=&{r3},0"(float %f)
  ret float %1
}

; Check support for returning a double in specific GPR with matching double
; input with hard float ABI
define double @dbl_gpr_matching_spec_reg_in_op_hard(double %d) #1 {
; CHECK-LABEL: dbl_gpr_matching_spec_reg_in_op_hard
; CHECK: vmov r2, r3, d0
; CHECK: mov r3, r2
; CHECK: vmov d0, r2, r3
  %1 = call double asm "mov ${0:R}, ${1:Q}", "=&{r2},0"(double %d)
  ret double %1
}

attributes #1 = { nounwind "target-features"="+d16,+vfp2,+vfp3,-fp-only-sp" "use-soft-float"="false" }
