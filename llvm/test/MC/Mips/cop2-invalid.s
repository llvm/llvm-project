# RUN: not llvm-mc %s -triple=mips-unknown-linux-gnu -mcpu=mips1 2>&1 \
# RUN:   | FileCheck %s
# RUN: not llvm-mc %s -triple=mips64-unknown-linux-gnu -mcpu=mips3 2>&1 \
# RUN:   | FileCheck %s
# RUN: not llvm-mc %s -triple=mips-unknown-linux-gnu -mcpu=mips32r6 2>&1 \
# RUN:   | FileCheck %s

  cop2 0x2000000   # CHECK: :[[@LINE]]:8: error: expected 25-bit unsigned immediate
  cop2 -1          # CHECK: :[[@LINE]]:8: error: expected 25-bit unsigned immediate
  cop2 sym         # CHECK: :[[@LINE]]:8: error: expected 25-bit unsigned immediate
  cop2             # CHECK: :[[@LINE]]:3: error: too few operands for instruction

  .set micromips
  cop2 0           # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
