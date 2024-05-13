# RUN: not llvm-mc -triple riscv64 -mattr=+experimental-zalasr < %s 2>&1 | FileCheck -check-prefixes=CHECK %s

# CHECK: error: unrecognized instruction mnemonic
lw. a1, (t0)

# CHECK: error: unrecognized instruction mnemonic
lw.rl t3, 0(t5)

# CHECK: error: unrecognized instruction mnemonic
lh.rlaq t4, (t6)

# CHECK: error: unrecognized instruction mnemonic
sb. a1, (t0)

# CHECK: error: unrecognized instruction mnemonic
sh.aq t3, 0(t5)

# CHECK: error: unrecognized instruction mnemonic
sh.rlaq t4, (t6)

# CHECK: error: optional integer offset must be 0
lw.aq zero, 1(a0)

# CHECK: error: optional integer offset must be 0
sw.rl t1, 2(s0)

# CHECK: error: optional integer offset must be 0
sb.aqrl sp, 3(s2)
