# RUN: not llvm-mc -triple=riscv32 --mattr=+experimental-p %s 2>&1 \
# RUN:     | FileCheck %s

# Imm overflow
pli.h a0, 0x400 # CHECK: :[[@LINE]]:11: error: immediate must be an integer in the range [-512, 511]
plui.h a1, 0x400 # CHECK: :[[@LINE]]:12: error: immediate must be an integer in the range [-512, 1023]
pli.b a0, 0x200 # CHECK: :[[@LINE]]:11: error: immediate must be an integer in the range [0, 255]

pslli.b a6, a7, 100 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 7]
pslli.h ra, sp, 100 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 15]
pslli.w ra, sp, 12 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set

psslai.h t0, t1, 100 # CHECK: :[[@LINE]]:18: error: immediate must be an integer in the range [0, 15]
psslai.w t0, t1, 27 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
sslai a4, a5, -1 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 31]

psll.ws a0, a1, a2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
padd.ws a0, a1, a2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
pssha.ws a0, a1, a2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
psshar.ws a0, a1, a2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
sha a0, a1, a2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
shar a0, a1, a2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set

psrli.b a6, a7, 100 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 7]
psrli.h ra, sp, 100 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 15]
psrli.w ra, sp, 31 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set

pusati.h ra, sp, 100 # CHECK: :[[@LINE]]:18: error: immediate must be an integer in the range [0, 15]
pusati.w ra, sp, 0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
usati ra, sp, 100 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 31]

psrai.b a6, a7, 100 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 7]
psrai.h ra, sp, 100 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 15]
psrai.w ra, sp, 10 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set

psrari.h ra, sp, 100 # CHECK: :[[@LINE]]:18: error: immediate must be an integer in the range [0, 15]
psrari.w ra, sp, 15 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
srari ra, sp, 100 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 31]

psati.h ra, sp, 100 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 15]
psati.w ra, sp, 24 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
sati ra, sp, 100 # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 31]

psrl.ws a0, a1, a2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
predsum.ws a0, a1, a2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
predsumu.ws a0, a1, a2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
psra.ws a0, a1, a2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
