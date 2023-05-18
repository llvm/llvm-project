; RUN: llc -mtriple armv7-arm-linux-gnueabihf -O2 -mcpu=cortex-a7 < %s | FileCheck %s
; RUN: llc -mtriple armv7-arm-linux-gnueabihf -O2 -mcpu=cortex-a7 -early-live-intervals < %s | FileCheck %s

%struct.twofloat = type { float, float }
%struct.twodouble = type { double, double }

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

; Check support for returning several float in GPR
define arm_aapcscc float @zerobits_float_convoluted_soft() #0 {
; CHECK-LABEL: zerobits_float_convoluted_soft
; CHECK: mov r0, #0
; CHECK-NEXT: mov r1, #0
  %1 = call { float, float } asm "mov $0, #0; mov $1, #0", "=r,=r"()
  %asmresult = extractvalue { float, float } %1, 0
  %asmresult1 = extractvalue { float, float } %1, 1
  %add = fadd float %asmresult, %asmresult1
  ret float %add
}

; Check support for returning several double in GPR
define double @zerobits_double_convoluted_soft() #0 {
; CHECK-LABEL: zerobits_double_convoluted_soft
; CHECK: mov r0, #0
; CHECK-NEXT: mov r1, #0
; CHECK-NEXT: mov r2, #0
; CHECK-NEXT: mov r3, #0
  %1 = call { double, double } asm "mov ${0:Q}, #0; mov ${0:R}, #0; mov ${1:Q}, #0; mov ${1:R}, #0", "=r,=r"()
  %asmresult = extractvalue { double, double } %1, 0
  %asmresult1 = extractvalue { double, double } %1, 1
  %add = fadd double %asmresult, %asmresult1
  ret double %add
}

; Check support for returning several floats in GPRs with matching float inputs
; with soft float ABI
define arm_aapcscc float @flt_gprs_matching_in_op_soft(float %f1, float %f2) #0 {
; CHECK-LABEL: flt_gprs_matching_in_op_soft
; CHECK: mov r0, r0
; CHECK-NEXT: mov r1, r1
  %1 = call { float, float } asm "mov $0, $2; mov $1, $3", "=&r,=&r,0,1"(float %f1, float %f2)
  %asmresult1 = extractvalue { float, float } %1, 0
  %asmresult2 = extractvalue { float, float } %1, 1
  %add = fadd float %asmresult1, %asmresult2
  ret float %add
}

; Check support for returning several double in GPRs with matching double input
; with soft float ABI
define arm_aapcscc double @dbl_gprs_matching_in_op_soft(double %d1, double %d2) #0 {
; CHECK-LABEL: dbl_gprs_matching_in_op_soft
; CHECK: mov r1, r0
; CHECK-NEXT: mov r3, r2
  %1 = call { double, double } asm "mov ${0:R}, ${2:Q}; mov ${1:R}, ${3:Q}", "=&r,=&r,0,1"(double %d1, double %d2)
  %asmresult1 = extractvalue { double, double } %1, 0
  %asmresult2 = extractvalue { double, double } %1, 1
  %add = fadd double %asmresult1, %asmresult2
  ret double %add
}

; Check support for returning several float in specific GPRs with matching
; float input with soft float ABI
define arm_aapcscc float @flt_gprs_matching_spec_reg_in_op_soft(float %f1, float %f2) #0 {
; CHECK-LABEL: flt_gprs_matching_spec_reg_in_op_soft
; CHECK: mov r3, r3
; CHECK-NEXT: mov r4, r4
  %1 = call { float, float } asm "mov $0, $2; mov $1, $3", "=&{r3},=&{r4},0,1"(float %f1, float %f2)
  %asmresult1 = extractvalue { float, float } %1, 0
  %asmresult2 = extractvalue { float, float } %1, 1
  %add = fadd float %asmresult1, %asmresult2
  ret float %add
}

; Check support for returning several double in specific GPRs with matching
; double input with soft float ABI
define arm_aapcscc double @dbl_gprs_matching_spec_reg_in_op_soft(double %d1, double %d2) #0 {
; CHECK-LABEL: dbl_gprs_matching_spec_reg_in_op_soft
; CHECK: mov r3, r2
; CHECK-NEXT: mov r5, r4
  %1 = call { double, double } asm "mov ${0:R}, ${2:Q}; mov ${1:R}, ${3:Q}", "=&{r2},=&{r4},0,1"(double %d1, double %d2)
  %asmresult1 = extractvalue { double, double } %1, 0
  %asmresult2 = extractvalue { double, double } %1, 1
  %add = fadd double %asmresult1, %asmresult2
  ret double %add
}

attributes #0 = { nounwind "target-features"="-d32,+vfp2,+vfp3" "use-soft-float"="true" }


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

; Check support for returning several float in GPR
define %struct.twofloat @zerobits_float_convoluted_hard() #1 {
; CHECK-LABEL: zerobits_float_convoluted_hard
; CHECK: mov r0, #0
; CHECK-NEXT: mov r1, #0
; CHECK: vmov s0, r0
; CHECK-NEXT: vmov s1, r1
  %1 = call { float, float } asm "mov $0, #0; mov $1, #0", "=r,=r"()
  %asmresult1 = extractvalue { float, float } %1, 0
  %asmresult2 = extractvalue { float, float } %1, 1
  %partialres = insertvalue %struct.twofloat undef, float %asmresult1, 0
  %res = insertvalue %struct.twofloat %partialres, float %asmresult2, 1
  ret %struct.twofloat %res
}

; Check support for returning several double in GPR
define %struct.twodouble @zerobits_double_convoluted_hard() #1 {
; CHECK-LABEL: zerobits_double_convoluted_hard
; CHECK: mov r0, #0
; CHECK-NEXT: mov r1, #0
; CHECK-NEXT: mov r2, #0
; CHECK-NEXT: mov r3, #0
; CHECK: vmov d0, r0, r1
; CHECK-NEXT: vmov d1, r2, r3
  %1 = call { double, double } asm "mov ${0:Q}, #0; mov ${0:R}, #0; mov ${1:Q}, #0; mov ${1:R}, #0", "=r,=r"()
  %asmresult1 = extractvalue { double, double } %1, 0
  %asmresult2 = extractvalue { double, double } %1, 1
  %partialres = insertvalue %struct.twodouble undef, double %asmresult1, 0
  %res = insertvalue %struct.twodouble %partialres, double %asmresult2, 1
  ret %struct.twodouble %res
}

; Check support for returning several floats in GPRs with matching float inputs
; with hard float ABI
define %struct.twofloat @flt_gprs_matching_in_op_hard(float %f1, float %f2) #1 {
; CHECK-LABEL: flt_gprs_matching_in_op_hard
; CHECK: vmov r0, s0
; CHECK-NEXT: vmov r1, s1
; CHECK: mov r0, r0
; CHECK-NEXT: mov r1, r1
; CHECK: vmov s0, r0
; CHECK-NEXT: vmov s1, r1
  %1 = call { float, float } asm "mov $0, $2; mov $1, $3", "=&r,=&r,0,1"(float %f1, float %f2)
  %asmresult1 = extractvalue { float, float } %1, 0
  %asmresult2 = extractvalue { float, float } %1, 1
  %partialres = insertvalue %struct.twofloat undef, float %asmresult1, 0
  %res = insertvalue %struct.twofloat %partialres, float %asmresult2, 1
  ret %struct.twofloat %res
}

; Check support for returning several double in GPRs with matching double input
; with hard float ABI
define %struct.twodouble @dbl_gprs_matching_in_op_hard(double %d1, double %d2) #1 {
; CHECK-LABEL: dbl_gprs_matching_in_op_hard
; CHECK: vmov r0, r1, d0
; CHECK-NEXT: vmov r2, r3, d1
; CHECK: mov r1, r0
; CHECK-NEXT: mov r3, r2
; CHECK: vmov d0, r0, r1
; CHECK-NEXT: vmov d1, r2, r3
  %1 = call { double, double } asm "mov ${0:R}, ${2:Q}; mov ${1:R}, ${3:Q}", "=&r,=&r,0,1"(double %d1, double %d2)
  %asmresult1 = extractvalue { double, double } %1, 0
  %asmresult2 = extractvalue { double, double } %1, 1
  %partialres = insertvalue %struct.twodouble undef, double %asmresult1, 0
  %res = insertvalue %struct.twodouble %partialres, double %asmresult2, 1
  ret %struct.twodouble %res
}

; Check support for returning several float in specific GPRs with matching
; float input with hard float ABI
define %struct.twofloat @flt_gprs_matching_spec_reg_in_op_hard(float %f1, float %f2) #1 {
; CHECK-LABEL: flt_gprs_matching_spec_reg_in_op_hard
; CHECK: vmov r3, s0
; CHECK-NEXT: vmov r4, s1
; CHECK: mov r3, r3
; CHECK-NEXT: mov r4, r4
; CHECK: vmov s0, r3
; CHECK-NEXT: vmov s1, r4
  %1 = call { float, float } asm "mov $0, $2; mov $1, $3", "=&{r3},=&{r4},0,1"(float %f1, float %f2)
  %asmresult1 = extractvalue { float, float } %1, 0
  %asmresult2 = extractvalue { float, float } %1, 1
  %partialres = insertvalue %struct.twofloat undef, float %asmresult1, 0
  %res = insertvalue %struct.twofloat %partialres, float %asmresult2, 1
  ret %struct.twofloat %res
}

; Check support for returning several double in specific GPRs with matching
; double input with hard float ABI
define %struct.twodouble @dbl_gprs_matching_spec_reg_in_op_hard(double %d1, double %d2) #1 {
; CHECK-LABEL: dbl_gprs_matching_spec_reg_in_op_hard
; CHECK: vmov r2, r3, d0
; CHECK-NEXT: vmov r4, r5, d1
; CHECK: mov r3, r2
; CHECK-NEXT: mov r5, r4
; CHECK: vmov d0, r2, r3
; CHECK-NEXT: vmov d1, r4, r5
  %1 = call { double, double } asm "mov ${0:R}, ${2:Q}; mov ${1:R}, ${3:Q}", "=&{r2},=&{r4},0,1"(double %d1, double %d2)
  %asmresult1 = extractvalue { double, double } %1, 0
  %asmresult2 = extractvalue { double, double } %1, 1
  %partialres = insertvalue %struct.twodouble undef, double %asmresult1, 0
  %res = insertvalue %struct.twodouble %partialres, double %asmresult2, 1
  ret %struct.twodouble %res
}

attributes #1 = { nounwind "target-features"="-d32,+vfp2,+vfp3" "use-soft-float"="false" }
