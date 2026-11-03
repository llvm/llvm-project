# RUN: llvm-mc -triple=i386-unknown-linux-gnu -mattr=+lvi-cfi -x86-experimental-lvi-inline-asm-hardening -show-encoding %s 2>&1 | FileCheck %s

.code16
ret

# CHECK: warning: Instruction may be vulnerable to LVI and requires manual mitigation
# CHECK: note: See https://software.intel.com/security-software-guidance/insights/deep-dive-load-value-injection#specialinstructions for more information
# CHECK: retw
