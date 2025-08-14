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
