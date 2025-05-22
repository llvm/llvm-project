llvm/test/tools/llvm-exegesis/AArch64/skip_unsupported_instructions.s

# REQUIRES: aarch64-registered-target

# Check for skipping of illegal instruction errors (AUT and LDGM)
# RUN: llvm-exegesis -mcpu=neoverse-v2 -mode=latency --opcode-name=AUTIA --benchmark-phase=assemble-measured-code 2>&1 | FileCheck %s --check-prefix=CHECK-AUTIA
# CHECK-AUTIA-NOT: snippet crashed while running: Illegal instruction

# RUN: llvm-exegesis -mcpu=neoverse-v2 -mode=latency --opcode-name=LDGM --benchmark-phase=assemble-measured-code 2>&1 | FileCheck %s --check-prefix=CHECK-LDGM
# CHECK-LDGM: LDGM: Unsupported opcode: load tag multiple 