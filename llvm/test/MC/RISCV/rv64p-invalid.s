# RUN: llvm-mc --verify -triple=riscv64 --mattr=+experimental-p %s

# Imm overflow
pli.h a0, 0x400 # expected-error:11 {{immediate must be an integer in the range [-512, 511]}}
plui.h a1, 0x400 # expected-error:12 {{immediate must be an integer in the range [-512, 1023]}}
pli.w a1, -0x201 # expected-error:11 {{immediate must be an integer in the range [-512, 511]}}

pslli.b a6, a7, 100 # expected-error:17 {{immediate must be an integer in the range [0, 7]}}
pslli.h ra, sp, 100 # expected-error:17 {{immediate must be an integer in the range [0, 15]}}
pslli.w ra, sp, 100 # expected-error:17 {{immediate must be an integer in the range [0, 31]}}

ssha a0, a1, a2 # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}
sshar a0, a1, a2 # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}

psslai.h t0, t1, 100 # expected-error:18 {{immediate must be an integer in the range [0, 15]}}
psslai.w a4, a5, -1 # expected-error:18 {{immediate must be an integer in the range [0, 31]}}
sslai ra, sp, 10 # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}

psrli.b a6, a7, 100 # expected-error:17 {{immediate must be an integer in the range [0, 7]}}
psrli.h ra, sp, 100 # expected-error:17 {{immediate must be an integer in the range [0, 15]}}
psrli.w ra, sp, 100 # expected-error:17 {{immediate must be an integer in the range [0, 31]}}

pusati.h ra, sp, 100 # expected-error:18 {{immediate must be an integer in the range [0, 15]}}
pusati.w ra, sp, 100 # expected-error:18 {{immediate must be an integer in the range [0, 31]}}
usati ra, sp, 100 # expected-error:15 {{immediate must be an integer in the range [0, 63]}}

psrai.b a6, a7, 100 # expected-error:17 {{immediate must be an integer in the range [0, 7]}}
psrai.h ra, sp, 100 # expected-error:17 {{immediate must be an integer in the range [0, 15]}}
psrai.w ra, sp, 100 # expected-error:17 {{immediate must be an integer in the range [0, 31]}}

psrari.h ra, sp, 100 # expected-error:18 {{immediate must be an integer in the range [0, 15]}}
psrari.w ra, sp, 100 # expected-error:18 {{immediate must be an integer in the range [0, 31]}}
srari ra, sp, 100 # expected-error:15 {{immediate must be an integer in the range [0, 63]}}

psati.h ra, sp, 100 # expected-error:17 {{immediate must be an integer in the range [1, 16]}}
psati.w ra, sp, 100 # expected-error:17 {{immediate must be an integer in the range [1, 32]}}
sati ra, sp, 100 # expected-error:14 {{immediate must be an integer in the range [1, 64]}}

sadd t1, a7, a0 # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}
aadd t1, a7, a0 # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}
saddu t1, a7, a0 # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}
aaddu t1, a7, a0 # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}
ssub t1, a7, a0 # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}
asub t1, a7, a0 # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}
ssubu t1, a7, a0 # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}
asubu t1, a7, a0 # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}

mul.h01 t0, t1, t2 # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}
macc.h01 t3, t4, t5 # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}
mulu.h01 t0, t1, t2 # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}
maccu.h01 t3, t4, t5 # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}

ssh1sadd t0, t1, t2 # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}

mul.h00 a4, t4, s2 # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}
macc.h00 a4, a0, s0 # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}
mul.h11 a0, a4, a0 # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}
macc.h11 s6, a4, s4 # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}
mulu.h00 s6, s0, a0 # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}
maccu.h00 s0, s6, s0 # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}
mulu.h11 s0, s4, s6 # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}
maccu.h11 s0, t4, t4 # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}
mulsu.h00 a4, s4, s6 # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}
maccsu.h00 s4, s4, s0 # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}
mulsu.h11 s8, s4, s0 # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}
maccsu.h11 s0, a2, s6 # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}

# FIXME: This error doesn't make sense. Should say that we need RV32I.
pli.dh a0, 1 # expected-error:1 {{invalid instruction}}
pli.db s0, 1 # expected-error:1 {{invalid instruction}}
plui.dh t1, 1 # expected-error:1 {{invalid instruction}}
