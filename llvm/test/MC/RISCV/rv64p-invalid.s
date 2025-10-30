# RUN: not llvm-mc -triple=riscv64 --mattr=+experimental-p %s 2>&1 \
# RUN:     | FileCheck %s

# Imm overflow
pli.h a0, 0x400 # CHECK: :[[@LINE]]:11: error: immediate must be an integer in the range [-512, 511]
plui.h a1, 0x400 # CHECK: :[[@LINE]]:12: error: immediate must be an integer in the range [-512, 1023]
pli.w a1, -0x201 # CHECK: :[[@LINE]]:11: error: immediate must be an integer in the range [-512, 511]

pslli.b a6, a7, 100 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 7]
pslli.h ra, sp, 100 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 15]
pslli.w ra, sp, 100 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 31]

ssha a0, a1, a2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set
sshar a0, a1, a2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set

psslai.h t0, t1, 100 # CHECK: :[[@LINE]]:18: error: immediate must be an integer in the range [0, 15]
psslai.w a4, a5, -1 # CHECK: :[[@LINE]]:18: error: immediate must be an integer in the range [0, 31]
sslai ra, sp, 10 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set

psrli.b a6, a7, 100 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 7]
psrli.h ra, sp, 100 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 15]
psrli.w ra, sp, 100 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 31]

pusati.h ra, sp, 100 # CHECK: :[[@LINE]]:18: error: immediate must be an integer in the range [0, 15]
pusati.w ra, sp, 100 # CHECK: :[[@LINE]]:18: error: immediate must be an integer in the range [0, 31]
usati ra, sp, 100 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 63]

psrai.b a6, a7, 100 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 7]
psrai.h ra, sp, 100 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 15]
psrai.w ra, sp, 100 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 31]

psrari.h ra, sp, 100 # CHECK: :[[@LINE]]:18: error: immediate must be an integer in the range [0, 15]
psrari.w ra, sp, 100 # CHECK: :[[@LINE]]:18: error: immediate must be an integer in the range [0, 31]
srari ra, sp, 100 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 63]

psati.h ra, sp, 100 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 15]
psati.w ra, sp, 100 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 31]
sati ra, sp, 100 # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 63]

sadd t1, a7, a0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set
aadd t1, a7, a0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set
saddu t1, a7, a0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set
aaddu t1, a7, a0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set
ssub t1, a7, a0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set
asub t1, a7, a0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set
ssubu t1, a7, a0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set
asubu t1, a7, a0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set

mul.h01 t0, t1, t2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set
macc.h01 t3, t4, t5 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set
mulu.h01 t0, t1, t2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set
maccu.h01 t3, t4, t5 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set

ssh1sadd t0, t1, t2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set

mul.h00 a4, t4, s2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set
macc.h00 a4, a0, s0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set
mul.h11 a0, a4, a0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set
macc.h11 s6, a4, s4 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set
mulu.h00 s6, s0, a0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set
maccu.h00 s0, s6, s0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set
mulu.h11 s0, s4, s6 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set
maccu.h11 s0, t4, t4 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set
mulsu.h00 a4, s4, s6 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set
maccsu.h00 s4, s4, s0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set
mulsu.h11 s8, s4, s0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set
maccsu.h11 s0, a2, s6 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set

# FIXME: This error doesn't make sense. Should say that we need RV32I.
pli.dh a0, 1 # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
pli.db s0, 1 # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
plui.dh t1, 1 # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
