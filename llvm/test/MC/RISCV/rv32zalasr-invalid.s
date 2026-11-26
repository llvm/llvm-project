# RUN: llvm-mc --verify -triple riscv32 -mattr=+zalasr < %s

ld.aq a1, (t0) # expected-error {{instruction requires the following: RV64I Base Instruction Set}}

ld.aqrl a1, (t0) # expected-error {{instruction requires the following: RV64I Base Instruction Set}}

sd.rl a1, (t0) # expected-error {{instruction requires the following: RV64I Base Instruction Set}}

sd.aqrl a1, (t0) # expected-error {{instruction requires the following: RV64I Base Instruction Set}}

lw. a1, (t0) # expected-error {{unrecognized instruction mnemonic}}

lw.rl t3, 0(t5) # expected-error {{unrecognized instruction mnemonic}}

lh.rlaq t4, (t6) # expected-error {{unrecognized instruction mnemonic}}

sb. a1, (t0) # expected-error {{unrecognized instruction mnemonic}}

sh.aq t3, 0(t5) # expected-error {{unrecognized instruction mnemonic}}

sh.rlaq t4, (t6) # expected-error {{unrecognized instruction mnemonic}}

lw.aq zero, 1(a0) # expected-error {{optional integer offset must be 0}}

sw.rl t1, 2(s0) # expected-error {{optional integer offset must be 0}}

sb.aqrl sp, 3(s2) # expected-error {{optional integer offset must be 0}}
