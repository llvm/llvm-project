# RUN: llvm-mc --verify -triple riscv64 -mattr=+zalasr < %s

lw. a1, (t0) # expected-error {{unrecognized instruction mnemonic}}

lw.rl t3, 0(t5) # expected-error {{unrecognized instruction mnemonic}}

lh.rlaq t4, (t6) # expected-error {{unrecognized instruction mnemonic}}

sb. a1, (t0) # expected-error {{unrecognized instruction mnemonic}}

sh.aq t3, 0(t5) # expected-error {{unrecognized instruction mnemonic}}

sh.rlaq t4, (t6) # expected-error {{unrecognized instruction mnemonic}}

lw.aq zero, 1(a0) # expected-error {{optional integer offset must be 0}}

sw.rl t1, 2(s0) # expected-error {{optional integer offset must be 0}}

sb.aqrl sp, 3(s2) # expected-error {{optional integer offset must be 0}}
