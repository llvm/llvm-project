# RUN: not llvm-mc -triple=riscv32 --mattr=+experimental-p %s 2>&1 \
# RUN:     | FileCheck %s

# Imm overflow
pli.h a0, 0x400 # CHECK: :[[@LINE]]:11: error: immediate must be an integer in the range [-512, 511]
plui.h a1, 0x400 # CHECK: :[[@LINE]]:12: error: immediate must be an integer in the range [-512, 1023]
pli.b a0, 0x200 # CHECK: :[[@LINE]]:11: error: immediate must be an integer in the range [-128, 255]

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

padd.w t3, s0, a0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
psadd.w t3, t1, s2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
paadd.w t5, t1, a4 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
psaddu.w s0, s2, t5 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
paaddu.w s0, t1, s0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
psub.w t3, a0, s0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
pssub.w t3, a4, t1 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
pasub.w t3, a2, a4 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
pssubu.w a0, a4, t3 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
pasubu.w a0, t3, a4 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set

pmul.w.h01 s2, t5, t3 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
mul.w01 t5, a4, t1 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
pmacc.w.h01 t1, t1, a0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
macc.w01 s2, a0, t3 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
pmulu.w.h01 t1, a4, a2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
mulu.w01 t5, t1, t5 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
pmaccu.w.h01 t5, t5, a4 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
maccu.w01 a0, a0, t1 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set

psh1add.w s2, t5, a2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
pssh1sadd.w a4, t3, s0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
unzip8p a4, t3, t1 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
unzip16p t5, a4, t3 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
unzip8hp s0, a0, t1 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
unzip16hp a0, a0, a2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
zip8p t5, t3, t3 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
zip16p a0, t5, a0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
zip8hp t5, a0, a2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
zip16hp t1, t5, a4 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set

pmul.w.h00 s2, t1, a2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
mul.w00 a4, a0, a2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
pmacc.w.h00 s2, t5, a2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
macc.w00 t1, a0, t5 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
pmul.w.h11 s0, a4, s0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
mul.w11 a0, t3, a0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
pmacc.w.h11 a4, a4, t1 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
macc.w11 t3, s2, a2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
pmulu.w.h00 a2, t3, t1 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
mulu.w00 a0, t5, s2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
pmaccu.w.h00 t3, t3, t5 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
maccu.w00 s2, t1, s2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
pmulu.w.h11 s0, t5, t5 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
mulu.w11 s0, t1, a4 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
pmaccu.w.h11 a0, s0, t5 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
maccu.w11 s2, t3, t5 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
pmulsu.w.h00 t5, t5, t5 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
mulsu.w00 t1, s0, a4 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
pmulsu.w.h11 t1, t3, s2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
mulsu.w11 a2, s2, a2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
pmaccsu.w.h00 a4, a0, t1 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
maccsu.w00 a4, s2, s0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
pmaccsu.w.h11 a0, a2, t3 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
maccsu.w11 t5, a4, s2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set

ppack.w t5, a2, a4 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
ppackbt.w t5, s0, t5 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
ppacktb.w t5, t1, t1 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
ppackt.w t3, a0, s2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set

pli.dh a1, 1 # CHECK: :[[@LINE]]:8: error: register must be even
pli.db s1, 1 # CHECK: :[[@LINE]]:8: error: register must be even
plui.dh t2, 1 # CHECK: :[[@LINE]]:9: error: register must be even

pli.dh a0, 0x400 # CHECK: :[[@LINE]]:12: error: immediate must be an integer in the range [-512, 511]
pli.db a0, 0x200 # CHECK: :[[@LINE]]:12: error: immediate must be an integer in the range [-128, 255]
plui.dh a0, 0x400 # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [-512, 1023]
