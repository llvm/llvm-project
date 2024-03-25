## Some section flags are processor-specific. Reject them for other targets.
# REQUIRES: riscv-registered-target
# RUN: not llvm-mc -triple=riscv32 %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

# CHECK: {{.*}}.s:[[# @LINE+1]]:34: error: unknown flag
.section XCORE_SHF_CP_SECTION,"c",@progbits

# CHECK: {{.*}}.s:[[# @LINE+1]]:34: error: unknown flag
.section XCORE_SHF_CP_SECTION,"d",@progbits

# CHECK: {{.*}}.s:[[# @LINE+1]]:27: error: unknown flag
.section SHF_HEX_GPREL,"s",@progbits

# CHECK: {{.*}}.s:[[# @LINE+1]]:30: error: unknown flag
.section SHF_ARM_PURECODE,"y",@progbits

# CHECK: {{.*}}.s:[[# @LINE+1]]:30: error: unknown flag
.section SHF_X86_64_LARGE,"l",@progbits
