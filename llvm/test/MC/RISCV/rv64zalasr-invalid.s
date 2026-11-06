# RUN: llvm-mc --verify -triple riscv64 -mattr=+zalasr < %s

# expected-error@+1 {{unrecognized instruction mnemonic}}
lw. a1, (t0)

# expected-error@+1 {{unrecognized instruction mnemonic}}
lw.rl t3, 0(t5)

# expected-error@+1 {{unrecognized instruction mnemonic}}
lh.rlaq t4, (t6)

# expected-error@+1 {{unrecognized instruction mnemonic}}
sb. a1, (t0)

# expected-error@+1 {{unrecognized instruction mnemonic}}
sh.aq t3, 0(t5)

# expected-error@+1 {{unrecognized instruction mnemonic}}
sh.rlaq t4, (t6)

# expected-error@+1 {{optional integer offset must be 0}}
lw.aq zero, 1(a0)

# expected-error@+1 {{optional integer offset must be 0}}
sw.rl t1, 2(s0)

# expected-error@+1 {{optional integer offset must be 0}}
sb.aqrl sp, 3(s2)
