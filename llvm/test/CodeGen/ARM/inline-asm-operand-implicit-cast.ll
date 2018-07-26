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

attributes #1 = { nounwind "target-features"="+d16,+vfp2,+vfp3,-fp-only-sp" "use-soft-float"="false" }
