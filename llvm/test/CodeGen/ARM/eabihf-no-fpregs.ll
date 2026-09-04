; RUN: not llc --mtriple=armv7-none-eabi --mattr=-fpregs < %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:
; RUN: not llc --mtriple=armv7-none-eabihf --mattr=-fpregs < %s -o /dev/null 2>&1 | FileCheck %s --check-prefixes=CHECK,EABIHF --implicit-check-not=error:
; RUN: not llc --mtriple=thumbv6-none-eabihf --mcpu=arm1176jzf-s < %s -o /dev/null 2>&1 | FileCheck %s --check-prefixes=CHECK,EABIHF --implicit-check-not=error:

; EABIHF: error: <unknown>:0:0: in function default_pcs void (): calling convention is hard-float, but floating-point registers are unavailable
define void @default_pcs() {
  ret void
}

; CHECK: error: {{.*}} in function hard_pcs {{.*}}: calling convention is hard-float, but floating-point registers are unavailable
define arm_aapcs_vfpcc void @hard_pcs() {
  ret void
}

define arm_aapcscc void @soft_pcs() {
  ret void
}

define void @variadic(...) {
  ret void
}

; CHECK: error: {{.*}} in function soft_to_hard {{.*}}: 'soft_to_hard' calls 'hard_callee', which expects a hard-float calling convention, but floating-point registers are unavailable
; CHECK: error: {{.*}} 'soft_to_hard' calls 'hard_callee2', which expects a hard-float calling convention, but floating-point registers are unavailable
define arm_aapcscc void @soft_to_hard() {
  call arm_aapcs_vfpcc void @hard_callee()
  call arm_aapcs_vfpcc void @hard_callee2()
  ret void
}

; EABIHF: error: {{.*}} in function soft_to_default_hard {{.*}}: 'soft_to_default_hard' calls 'default_callee', which expects a hard-float calling convention, but floating-point registers are unavailable
define arm_aapcscc void @soft_to_default_hard() {
  call void @default_callee()
  ret void
}

declare arm_aapcs_vfpcc void @hard_callee()
declare arm_aapcs_vfpcc void @hard_callee2()
declare void @default_callee()
