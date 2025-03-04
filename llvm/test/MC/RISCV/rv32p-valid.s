# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-p -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-p < %s \
# RUN:     | llvm-objdump --mattr=+experimental-p -M no-aliases -d -r --no-print-imm-hex - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: clz a0, a1
# CHECK-ASM: encoding: [0x13,0x95,0x05,0x60]
clz a0, a1
# CHECK-ASM-AND-OBJ: cls a1, a2
# CHECK-ASM: encoding: [0x93,0x15,0x36,0x60]
cls a1, a2
# CHECK-ASM-AND-OBJ: sext.b a2, a3
# CHECK-ASM: encoding: [0x13,0x96,0x46,0x60]
sext.b a2, a3
# CHECK-ASM-AND-OBJ: sext.h t0, t1
# CHECK-ASM: encoding: [0x93,0x12,0x53,0x60]
sext.h t0, t1
# CHECK-ASM-AND-OBJ: abs a4, a5
# CHECK-ASM: encoding: [0x13,0x97,0x77,0x60]
abs a4, a5
# CHECK-ASM-AND-OBJ: rev8 s0, s1
# CHECK-ASM: encoding: [0x13,0xd4,0x84,0x69]
rev8 s0, s1
# CHECK-ASM-AND-OBJ: rev s2, s3
# CHECK-ASM: encoding: [0x13,0xd9,0xf9,0x69]
rev s2, s3
# CHECK-ASM-AND-OBJ: sh1add a0, a1, a2
# CHECK-ASM: encoding: [0x33,0xa5,0xc5,0x20]
sh1add a0, a1, a2
# CHECK-ASM-AND-OBJ: pack s0, s1, s2
# CHECK-ASM: encoding: [0x33,0xc4,0x24,0x09]
pack s0, s1, s2
# CHECK-ASM-AND-OBJ: min t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x42,0x73,0x0a]
min t0, t1, t2
# CHECK-ASM-AND-OBJ: minu ra, sp, gp
# CHECK-ASM: encoding: [0xb3,0x50,0x31,0x0a]
minu ra, sp, gp
# CHECK-ASM-AND-OBJ: max t3, t4, t5
# CHECK-ASM: encoding: [0x33,0xee,0xee,0x0b]
max t3, t4, t5
# CHECK-ASM-AND-OBJ: maxu a4, a5, a6
# CHECK-ASM: encoding: [0x33,0xf7,0x07,0x0b]
maxu a4, a5, a6
# CHECK-ASM-AND-OBJ: pslli.b a6, a7
# CHECK-ASM: encoding: [0x1b,0xa8,0x88,0x80]
pslli.b a6, a7
# CHECK-ASM-AND-OBJ: pslli.h ra, sp
# CHECK-ASM: encoding: [0x9b,0x20,0x01,0x81]
pslli.h ra, sp
# CHECK-ASM-AND-OBJ: psslai.h t0, t1
# CHECK-ASM: encoding: [0x9b,0x22,0x03,0xd1]
psslai.h t0, t1
# CHECK-ASM-AND-OBJ: sslai a4, a5
# CHECK-ASM: encoding: [0x1b,0xa7,0x07,0xd2]
sslai a4, a5
# CHECK-ASM-AND-OBJ: pli.h a5, 16
# CHECK-ASM: encoding: [0x9b,0x27,0x08,0xb0]
pli.h a5, 16
# CHECK-ASM-AND-OBJ: pli.b a6, 16
# CHECK-ASM: encoding: [0x1b,0x28,0x10,0xb4]
pli.b a6, 16
# CHECK-ASM-AND-OBJ: psext.h.b a7, a0
# CHECK-ASM: encoding: [0x9b,0x28,0x45,0xe0]
psext.h.b a7, a0
# CHECK-ASM-AND-OBJ: psabs.h a1, a2
# CHECK-ASM: encoding: [0x9b,0x25,0x76,0xe0]
psabs.h a1, a2
# CHECK-ASM-AND-OBJ: psabs.b t0, t1
# CHECK-ASM: encoding: [0x9b,0x22,0x73,0xe2]
psabs.b t0, t1
# CHECK-ASM-AND-OBJ: plui.h gp, 32
# CHECK-ASM: encoding: [0x9b,0x21,0x10,0xf0]
plui.h gp, 32
# CHECK-ASM-AND-OBJ: psll.hs a0, a1, a2
# CHECK-ASM: encoding: [0x1b,0xa5,0xc5,0x88]
psll.hs a0, a1, a2
# CHECK-ASM-AND-OBJ: psll.bs a3, a4, a5
# CHECK-ASM: encoding: [0x9b,0x26,0xf7,0x8c]
psll.bs a3, a4, a5
# CHECK-ASM-AND-OBJ: padd.hs t0, t1, t2
# CHECK-ASM: encoding: [0x9b,0x22,0x73,0x98]
padd.hs t0, t1, t2
# CHECK-ASM-AND-OBJ: padd.bs ra, a1, a2
# CHECK-ASM: encoding: [0x9b,0xa0,0xc5,0x9c]
padd.bs ra, a1, a2
# CHECK-ASM-AND-OBJ: pssha.hs a3, a4, a5
# CHECK-ASM: encoding: [0x9b,0x26,0xf7,0xe8]
pssha.hs a3, a4, a5
# CHECK-ASM-AND-OBJ: ssha gp, a4, a5
# CHECK-ASM: encoding: [0x9b,0x21,0xf7,0xea]
ssha gp, a4, a5
# CHECK-ASM-AND-OBJ: psshar.hs a6, a7, a0
# CHECK-ASM: encoding: [0x1b,0xa8,0xa8,0xf8]
psshar.hs a6, a7, a0
# CHECK-ASM-AND-OBJ: sshar t1, a7, a0
# CHECK-ASM: encoding: [0x1b,0xa3,0xa8,0xfa]
sshar t1, a7, a0
# CHECK-ASM-AND-OBJ: psrli.b a1, a2
# CHECK-ASM: encoding: [0x9b,0x45,0x86,0x80]
psrli.b a1, a2
# CHECK-ASM-AND-OBJ: psrli.h a0, a1
# CHECK-ASM: encoding: [0x1b,0xc5,0x05,0x81]
psrli.h a0, a1
# CHECK-ASM-AND-OBJ: pusati.h a2, t1
# CHECK-ASM: encoding: [0x1b,0x46,0x03,0xa1]
pusati.h a2, t1
# CHECK-ASM-AND-OBJ: usati a3, t2
# CHECK-ASM: encoding: [0x9b,0xc6,0x03,0xa2]
usati a3, t2
# CHECK-ASM-AND-OBJ: psrai.b a4, a5
# CHECK-ASM: encoding: [0x1b,0xc7,0x87,0xc0]
psrai.b a4, a5
# CHECK-ASM-AND-OBJ: psrai.h a6, a7
# CHECK-ASM: encoding: [0x1b,0xc8,0x08,0xc1]
psrai.h a6, a7
# CHECK-ASM-AND-OBJ: psrari.h a0, a1
# CHECK-ASM: encoding: [0x1b,0xc5,0x05,0xd1]
psrari.h a0, a1
# CHECK-ASM-AND-OBJ: srari a2, a3
# CHECK-ASM: encoding: [0x1b,0xc6,0x06,0xd2]
srari a2, a3
# CHECK-ASM-AND-OBJ: psati.h a4, t0
# CHECK-ASM: encoding: [0x1b,0xc7,0x02,0xe1]
psati.h a4, t0
# CHECK-ASM-AND-OBJ: sati a5, t1
# CHECK-ASM: encoding: [0x9b,0x47,0x03,0xe2]
sati a5, t1
# CHECK-ASM-AND-OBJ: psrl.hs a6, a7, a1
# CHECK-ASM: encoding: [0x1b,0xc8,0xb8,0x88]
psrl.hs a6, a7, a1
# CHECK-ASM-AND-OBJ: psrl.bs a1, a2, a3
# CHECK-ASM: encoding: [0x9b,0x45,0xd6,0x8c]
psrl.bs a1, a2, a3
# CHECK-ASM-AND-OBJ: predsum.hs a4, a5, a6
# CHECK-ASM: encoding: [0x1b,0xc7,0x07,0x99]
predsum.hs a4, a5, a6
# CHECK-ASM-AND-OBJ: predsum.bs a7, a1, a1
# CHECK-ASM: encoding: [0x9b,0xc8,0xb5,0x9c]
predsum.bs a7, a1, a1
# CHECK-ASM-AND-OBJ: predsumu.hs t0, t1, t2
# CHECK-ASM: encoding: [0x9b,0x42,0x73,0xb8]
predsumu.hs t0, t1, t2
# CHECK-ASM-AND-OBJ: predsumu.bs t3, t4, t5
# CHECK-ASM: encoding: [0x1b,0xce,0xee,0xbd]
predsumu.bs t3, t4, t5
# CHECK-ASM-AND-OBJ: psra.hs ra, a1, a2
# CHECK-ASM: encoding: [0x9b,0xc0,0xc5,0xc8]
psra.hs ra, a1, a2
# CHECK-ASM-AND-OBJ: psra.bs sp, a2, a3
# CHECK-ASM: encoding: [0x1b,0x41,0xd6,0xcc]
psra.bs sp, a2, a3
# CHECK-ASM-AND-OBJ: psslai.h t0, t1
# CHECK-ASM: encoding: [0x9b,0x22,0x03,0xd1]
psslai.h t0, t1
# CHECK-ASM-AND-OBJ: sslai a4, a5
# CHECK-ASM: encoding: [0x1b,0xa7,0x07,0xd2]
sslai a4, a5
# CHECK-ASM-AND-OBJ: pli.h a5, 16
# CHECK-ASM: encoding: [0x9b,0x27,0x08,0xb0]
pli.h a5, 16
# CHECK-ASM-AND-OBJ: pli.b a6, 16
# CHECK-ASM: encoding: [0x1b,0x28,0x10,0xb4]
pli.b a6, 16
# CHECK-ASM-AND-OBJ: psext.h.b a7, a0
# CHECK-ASM: encoding: [0x9b,0x28,0x45,0xe0]
psext.h.b a7, a0
# CHECK-ASM-AND-OBJ: psabs.h a1, a2
# CHECK-ASM: encoding: [0x9b,0x25,0x76,0xe0]
psabs.h a1, a2
# CHECK-ASM-AND-OBJ: psabs.b t0, t1
# CHECK-ASM: encoding: [0x9b,0x22,0x73,0xe2]
psabs.b t0, t1
# CHECK-ASM-AND-OBJ: plui.h gp, 32
# CHECK-ASM: encoding: [0x9b,0x21,0x10,0xf0]
plui.h gp, 32
# CHECK-ASM-AND-OBJ: psll.hs a0, a1, a2
# CHECK-ASM: encoding: [0x1b,0xa5,0xc5,0x88]
psll.hs a0, a1, a2
# CHECK-ASM-AND-OBJ: psll.bs a3, a4, a5
# CHECK-ASM: encoding: [0x9b,0x26,0xf7,0x8c]
psll.bs a3, a4, a5
# CHECK-ASM-AND-OBJ: padd.hs t0, t1, t2
# CHECK-ASM: encoding: [0x9b,0x22,0x73,0x98]
padd.hs t0, t1, t2
# CHECK-ASM-AND-OBJ: padd.bs ra, a1, a2
# CHECK-ASM: encoding: [0x9b,0xa0,0xc5,0x9c]
padd.bs ra, a1, a2
# CHECK-ASM-AND-OBJ: pssha.hs a3, a4, a5
# CHECK-ASM: encoding: [0x9b,0x26,0xf7,0xe8]
pssha.hs a3, a4, a5
# CHECK-ASM-AND-OBJ: ssha gp, a4, a5
# CHECK-ASM: encoding: [0x9b,0x21,0xf7,0xea]
ssha gp, a4, a5
# CHECK-ASM-AND-OBJ: psshar.hs a6, a7, a0
# CHECK-ASM: encoding: [0x1b,0xa8,0xa8,0xf8]
psshar.hs a6, a7, a0
# CHECK-ASM-AND-OBJ: sshar t1, a7, a0
# CHECK-ASM: encoding: [0x1b,0xa3,0xa8,0xfa]
sshar t1, a7, a0
# CHECK-ASM-AND-OBJ: psrli.b a1, a2
# CHECK-ASM: encoding: [0x9b,0x45,0x86,0x80]
psrli.b a1, a2
# CHECK-ASM-AND-OBJ: psrli.h a0, a1
# CHECK-ASM: encoding: [0x1b,0xc5,0x05,0x81]
psrli.h a0, a1
# CHECK-ASM-AND-OBJ: pusati.h a2, t1
# CHECK-ASM: encoding: [0x1b,0x46,0x03,0xa1]
pusati.h a2, t1
# CHECK-ASM-AND-OBJ: usati a3, t2
# CHECK-ASM: encoding: [0x9b,0xc6,0x03,0xa2]
usati a3, t2
# CHECK-ASM-AND-OBJ: psrai.b a4, a5
# CHECK-ASM: encoding: [0x1b,0xc7,0x87,0xc0]
psrai.b a4, a5
# CHECK-ASM-AND-OBJ: psrai.h a6, a7
# CHECK-ASM: encoding: [0x1b,0xc8,0x08,0xc1]
psrai.h a6, a7
# CHECK-ASM-AND-OBJ: psrari.h a0, a1
# CHECK-ASM: encoding: [0x1b,0xc5,0x05,0xd1]
psrari.h a0, a1
# CHECK-ASM-AND-OBJ: srari a2, a3
# CHECK-ASM: encoding: [0x1b,0xc6,0x06,0xd2]
srari a2, a3
# CHECK-ASM-AND-OBJ: psati.h a4, t0
# CHECK-ASM: encoding: [0x1b,0xc7,0x02,0xe1]
psati.h a4, t0
# CHECK-ASM-AND-OBJ: sati a5, t1
# CHECK-ASM: encoding: [0x9b,0x47,0x03,0xe2]
sati a5, t1
# CHECK-ASM-AND-OBJ: psrl.hs a6, a7, a1
# CHECK-ASM: encoding: [0x1b,0xc8,0xb8,0x88]
psrl.hs a6, a7, a1
# CHECK-ASM-AND-OBJ: psrl.bs a1, a2, a3
# CHECK-ASM: encoding: [0x9b,0x45,0xd6,0x8c]
psrl.bs a1, a2, a3
# CHECK-ASM-AND-OBJ: predsum.hs a4, a5, a6
# CHECK-ASM: encoding: [0x1b,0xc7,0x07,0x99]
predsum.hs a4, a5, a6
# CHECK-ASM-AND-OBJ: predsum.bs a7, a1, a1
# CHECK-ASM: encoding: [0x9b,0xc8,0xb5,0x9c]
predsum.bs a7, a1, a1
# CHECK-ASM-AND-OBJ: predsumu.hs t0, t1, t2
# CHECK-ASM: encoding: [0x9b,0x42,0x73,0xb8]
predsumu.hs t0, t1, t2
# CHECK-ASM-AND-OBJ: predsumu.bs t3, t4, t5
# CHECK-ASM: encoding: [0x1b,0xce,0xee,0xbd]
predsumu.bs t3, t4, t5
# CHECK-ASM-AND-OBJ: psra.hs ra, a1, a2
# CHECK-ASM: encoding: [0x9b,0xc0,0xc5,0xc8]
psra.hs ra, a1, a2
# CHECK-ASM-AND-OBJ: psra.bs sp, a2, a3
# CHECK-ASM: encoding: [0x1b,0x41,0xd6,0xcc]
psra.bs sp, a2, a3
# CHECK-ASM-AND-OBJ: padd.h t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x02,0x73,0x80]
padd.h t0, t1, t2
# CHECK-ASM-AND-OBJ: padd.b ra, a1, a2
# CHECK-ASM: encoding: [0xbb,0x80,0xc5,0x84]
padd.b ra, a1, a2
# CHECK-ASM-AND-OBJ: psadd.h t3, t4, t5
# CHECK-ASM: encoding: [0x3b,0x8e,0xee,0x91]
psadd.h t3, t4, t5
# CHECK-ASM-AND-OBJ: sadd t1, a7, a0
# CHECK-ASM: encoding: [0x3b,0x83,0xa8,0x92]
sadd t1, a7, a0
# CHECK-ASM-AND-OBJ: psadd.b t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x02,0x73,0x94]
psadd.b t0, t1, t2
# CHECK-ASM-AND-OBJ: paadd.h t3, t4, t5
# CHECK-ASM: encoding: [0x3b,0x8e,0xee,0x99]
paadd.h t3, t4, t5
# CHECK-ASM-AND-OBJ: aadd t1, a7, a0
# CHECK-ASM: encoding: [0x3b,0x83,0xa8,0x9a]
aadd t1, a7, a0
# CHECK-ASM-AND-OBJ: paadd.b t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x02,0x73,0x9c]
paadd.b t0, t1, t2
# CHECK-ASM-AND-OBJ: psaddu.h t3, t4, t5
# CHECK-ASM: encoding: [0x3b,0x8e,0xee,0xb1]
psaddu.h t3, t4, t5
# CHECK-ASM-AND-OBJ: saddu t1, a7, a0
# CHECK-ASM: encoding: [0x3b,0x83,0xa8,0xb2]
saddu t1, a7, a0
# CHECK-ASM-AND-OBJ: psaddu.b t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x02,0x73,0xb4]
psaddu.b t0, t1, t2
# CHECK-ASM-AND-OBJ: paaddu.h t3, t4, t5
# CHECK-ASM: encoding: [0x3b,0x8e,0xee,0xb9]
paaddu.h t3, t4, t5
# CHECK-ASM-AND-OBJ: aaddu t1, a7, a0
# CHECK-ASM: encoding: [0x3b,0x83,0xa8,0xba]
aaddu t1, a7, a0
# CHECK-ASM-AND-OBJ: paaddu.b t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x02,0x73,0xbc]
paaddu.b t0, t1, t2
# CHECK-ASM-AND-OBJ: psub.h t3, t4, t5
# CHECK-ASM: encoding: [0x3b,0x8e,0xee,0xc1]
psub.h t3, t4, t5
# CHECK-ASM-AND-OBJ: psub.b t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x02,0x73,0xc4]
psub.b t0, t1, t2
# CHECK-ASM-AND-OBJ: pdif.h t3, t4, t5
# CHECK-ASM: encoding: [0x3b,0x8e,0xee,0xc9]
pdif.h t3, t4, t5
# CHECK-ASM-AND-OBJ: pdif.b t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x02,0x73,0xcc]
pdif.b t0, t1, t2
# CHECK-ASM-AND-OBJ: pssub.h t3, t4, t5
# CHECK-ASM: encoding: [0x3b,0x8e,0xee,0xd1]
pssub.h t3, t4, t5
# CHECK-ASM-AND-OBJ: ssub t1, a7, a0
# CHECK-ASM: encoding: [0x3b,0x83,0xa8,0xd2]
ssub t1, a7, a0
# CHECK-ASM-AND-OBJ: pssub.b t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x02,0x73,0xd4]
pssub.b t0, t1, t2
# CHECK-ASM-AND-OBJ: pasub.h t3, t4, t5
# CHECK-ASM: encoding: [0x3b,0x8e,0xee,0xd9]
pasub.h t3, t4, t5
# CHECK-ASM-AND-OBJ: asub t1, a7, a0
# CHECK-ASM: encoding: [0x3b,0x83,0xa8,0xda]
asub t1, a7, a0
# CHECK-ASM-AND-OBJ: pasub.b t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x02,0x73,0xdc]
pasub.b t0, t1, t2
# CHECK-ASM-AND-OBJ: pdifu.h t3, t4, t5
# CHECK-ASM: encoding: [0x3b,0x8e,0xee,0xe9]
pdifu.h t3, t4, t5
# CHECK-ASM-AND-OBJ: pdifu.b t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x02,0x73,0xec]
pdifu.b t0, t1, t2
# CHECK-ASM-AND-OBJ: pssubu.h t3, t4, t5
# CHECK-ASM: encoding: [0x3b,0x8e,0xee,0xf1]
pssubu.h t3, t4, t5
# CHECK-ASM-AND-OBJ: ssubu t1, a7, a0
# CHECK-ASM: encoding: [0x3b,0x83,0xa8,0xf2]
ssubu t1, a7, a0
# CHECK-ASM-AND-OBJ: pssubu.b t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x02,0x73,0xf4]
pssubu.b t0, t1, t2
# CHECK-ASM-AND-OBJ: pasubu.h t3, t4, t5
# CHECK-ASM: encoding: [0x3b,0x8e,0xee,0xf9]
pasubu.h t3, t4, t5
# CHECK-ASM-AND-OBJ: asubu t1, a7, a0
# CHECK-ASM: encoding: [0x3b,0x83,0xa8,0xfa]
asubu t1, a7, a0
# CHECK-ASM-AND-OBJ: pasubu.b t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x02,0x73,0xfc]
pasubu.b t0, t1, t2
# CHECK-ASM-AND-OBJ: slx gp, a4, a5
# CHECK-ASM: encoding: [0xbb,0x11,0xf7,0x8e]
slx gp, a4, a5
# CHECK-ASM-AND-OBJ: pmul.h.b01 t3, t4, t5
# CHECK-ASM: encoding: [0x3b,0x9e,0xee,0x91]
pmul.h.b01 t3, t4, t5
# CHECK-ASM-AND-OBJ: mul.h01 t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x12,0x73,0x92]
mul.h01 t0, t1, t2
# CHECK-ASM-AND-OBJ: macc.h01 t3, t4, t5
# CHECK-ASM: encoding: [0x3b,0x9e,0xee,0x9b]
macc.h01 t3, t4, t5
# CHECK-ASM-AND-OBJ: mvm a0, a1, a2
# CHECK-ASM: encoding: [0x3b,0x95,0xc5,0xa8]
mvm a0, a1, a2
# CHECK-ASM-AND-OBJ: mvmn gp, a4, a5
# CHECK-ASM: encoding: [0xbb,0x11,0xf7,0xaa]
mvmn gp, a4, a5
# CHECK-ASM-AND-OBJ: merge t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x12,0x73,0xac]
merge t0, t1, t2
# CHECK-ASM-AND-OBJ: srx gp, a4, a5
# CHECK-ASM: encoding: [0xbb,0x11,0xf7,0xae]
srx gp, a4, a5
# CHECK-ASM-AND-OBJ: pmulu.h.b01 t3, t4, t5
# CHECK-ASM: encoding: [0x3b,0x9e,0xee,0xb1]
pmulu.h.b01 t3, t4, t5
# CHECK-ASM-AND-OBJ: mulu.h01 t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x12,0x73,0xb2]
mulu.h01 t0, t1, t2
# CHECK-ASM-AND-OBJ: pdifsumu.b t3, t4, t5
# CHECK-ASM: encoding: [0x3b,0x9e,0xee,0xb5]
pdifsumu.b t3, t4, t5
# CHECK-ASM-AND-OBJ: maccu.h01 t3, t4, t5
# CHECK-ASM: encoding: [0x3b,0x9e,0xee,0xbb]
maccu.h01 t3, t4, t5
# CHECK-ASM-AND-OBJ: pdifsumau.b t3, t4, t5
# CHECK-ASM: encoding: [0x3b,0x9e,0xee,0xbd]
pdifsumau.b t3, t4, t5
# CHECK-ASM-AND-OBJ: mul.h01 t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x12,0x73,0x92]
mul.h01 t0, t1, t2
# CHECK-ASM-AND-OBJ: macc.h01 t3, t4, t5
# CHECK-ASM: encoding: [0x3b,0x9e,0xee,0x9b]
macc.h01 t3, t4, t5
# CHECK-ASM-AND-OBJ: mulu.h01 a0, a1, a2
# CHECK-ASM: encoding: [0x3b,0x95,0xc5,0xb2]
mulu.h01 a0, a1, a2
# CHECK-ASM-AND-OBJ: maccu.h01 a3, a4, a5
# CHECK-ASM: encoding: [0xbb,0x16,0xf7,0xba]
maccu.h01 a3, a4, a5
# CHECK-ASM-AND-OBJ: psh1add.h a0, a1, a2
# CHECK-ASM: encoding: [0x3b,0xa5,0xc5,0xa0]
psh1add.h a0, a1, a2
# CHECK-ASM-AND-OBJ: pssh1sadd.h a3, a4, a5
# CHECK-ASM: encoding: [0xbb,0x26,0xf7,0xb0]
pssh1sadd.h a3, a4, a5
# CHECK-ASM-AND-OBJ: ssh1sadd t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x22,0x73,0xa2]
ssh1sadd t0, t1, t2
# CHECK-ASM-AND-OBJ: pmul.h.b00 s6, t4, s8
# CHECK-ASM: encoding: [0x3b,0xbb,0x8e,0x81]
pmul.h.b00 s6, t4, s8
# CHECK-ASM-AND-OBJ: pmul.h.b11 t0, t4, s4
# CHECK-ASM: encoding: [0xbb,0xb2,0x4e,0x91]
pmul.h.b11 t0, t4, s4
# CHECK-ASM-AND-OBJ: pmulu.h.b00 t2, s6, s8
# CHECK-ASM: encoding: [0xbb,0x33,0x8b,0xa1]
pmulu.h.b00 t2, s6, s8
# CHECK-ASM-AND-OBJ: pmulu.h.b11 t2, s6, a0
# CHECK-ASM: encoding: [0xbb,0x33,0xab,0xb0]
pmulu.h.b11 t2, s6, a0
# CHECK-ASM-AND-OBJ: pmulsu.h.b00 s4, a4, s8
# CHECK-ASM: encoding: [0x3b,0x3a,0x87,0xe1]
pmulsu.h.b00 s4, a4, s8
# CHECK-ASM-AND-OBJ: pmulsu.h.b11 s6, a2, s4
# CHECK-ASM: encoding: [0x3b,0x3b,0x46,0xf1]
pmulsu.h.b11 s6, a2, s4
# CHECK-ASM-AND-OBJ: mul.h00 a4, t4, s2
# CHECK-ASM: encoding: [0x3b,0xb7,0x2e,0x83]
mul.h00 a4, t4, s2
# CHECK-ASM-AND-OBJ: macc.h00 a4, a0, s0
# CHECK-ASM: encoding: [0x3b,0x37,0x85,0x8a]
macc.h00 a4, a0, s0
# CHECK-ASM-AND-OBJ: mul.h11 a0, a4, a0
# CHECK-ASM: encoding: [0x3b,0x35,0xa7,0x92]
mul.h11 a0, a4, a0
# CHECK-ASM-AND-OBJ: macc.h11 s6, a4, s4
# CHECK-ASM: encoding: [0x3b,0x3b,0x47,0x9b]
macc.h11 s6, a4, s4
# CHECK-ASM-AND-OBJ: mulu.h00 s6, s0, a0
# CHECK-ASM: encoding: [0x3b,0x3b,0xa4,0xa2]
mulu.h00 s6, s0, a0
# CHECK-ASM-AND-OBJ: maccu.h00 s0, s6, s0
# CHECK-ASM: encoding: [0x3b,0x34,0x8b,0xaa]
maccu.h00 s0, s6, s0
# CHECK-ASM-AND-OBJ: mulu.h11 s0, s4, s6
# CHECK-ASM: encoding: [0x3b,0x34,0x6a,0xb3]
mulu.h11 s0, s4, s6
# CHECK-ASM-AND-OBJ: maccu.h11 s0, t4, t4
# CHECK-ASM: encoding: [0x3b,0xb4,0xde,0xbb]
maccu.h11 s0, t4, t4
# CHECK-ASM-AND-OBJ: mulsu.h00 a4, s4, s6
# CHECK-ASM: encoding: [0x3b,0x37,0x6a,0xe3]
mulsu.h00 a4, s4, s6
# CHECK-ASM-AND-OBJ: maccsu.h00 s4, s4, s0
# CHECK-ASM: encoding: [0x3b,0x3a,0x8a,0xea]
maccsu.h00 s4, s4, s0
# CHECK-ASM-AND-OBJ: mulsu.h11 s8, s4, s0
# CHECK-ASM: encoding: [0x3b,0x3c,0x8a,0xf2]
mulsu.h11 s8, s4, s0
# CHECK-ASM-AND-OBJ: maccsu.h11 s0, a2, s6
# CHECK-ASM: encoding: [0x3b,0x34,0x66,0xfb]
maccsu.h11 s0, a2, s6
# CHECK-ASM-AND-OBJ: ppack.h t1, a2, t5
# CHECK-ASM: encoding: [0x3b,0x43,0xe6,0x81]
ppack.h t1, a2, t5
# CHECK-ASM-AND-OBJ: ppackbt.h t5, t3, s2
# CHECK-ASM: encoding: [0x3b,0x4f,0x2e,0x91]
ppackbt.h t5, t3, s2
# CHECK-ASM-AND-OBJ: ppacktb.h t1, t1, s0
# CHECK-ASM: encoding: [0x3b,0x43,0x83,0xa0]
ppacktb.h t1, t1, s0
# CHECK-ASM-AND-OBJ: ppackt.h t3, s0, s0
# CHECK-ASM: encoding: [0x3b,0x4e,0x84,0xb0]
ppackt.h t3, s0, s0
# CHECK-ASM-AND-OBJ: packbt t1, t1, s2
# CHECK-ASM: encoding: [0x3b,0x43,0x23,0x93]
packbt t1, t1, s2
# CHECK-ASM-AND-OBJ: packtb t5, s0, a2
# CHECK-ASM: encoding: [0x3b,0x4f,0xc4,0xa2]
packtb t5, s0, a2
# CHECK-ASM-AND-OBJ: packt a4, t5, t5
# CHECK-ASM: encoding: [0x3b,0x47,0xef,0xb3]
packt a4, t5, t5
# CHECK-ASM-AND-OBJ: pm2add.h t3, t1, s0
# CHECK-ASM: encoding: [0x3b,0x5e,0x83,0x80]
pm2add.h t3, t1, s0
# CHECK-ASM-AND-OBJ: pm4add.b t1, s2, t5
# CHECK-ASM: encoding: [0x3b,0x53,0xe9,0x85]
pm4add.b t1, s2, t5
# CHECK-ASM-AND-OBJ: pm2adda.h t1, s2, a4
# CHECK-ASM: encoding: [0x3b,0x53,0xe9,0x88]
pm2adda.h t1, s2, a4
# CHECK-ASM-AND-OBJ: pm4adda.b t3, a0, t5
# CHECK-ASM: encoding: [0x3b,0x5e,0xe5,0x8d]
pm4adda.b t3, a0, t5
# CHECK-ASM-AND-OBJ: pm2add.hx s0, t5, a4
# CHECK-ASM: encoding: [0x3b,0x54,0xef,0x90]
pm2add.hx s0, t5, a4
# CHECK-ASM-AND-OBJ: pm2adda.hx a0, a0, t1
# CHECK-ASM: encoding: [0x3b,0x55,0x65,0x98]
pm2adda.hx a0, a0, t1
# CHECK-ASM-AND-OBJ: pm2addu.h s2, t5, a2
# CHECK-ASM: encoding: [0x3b,0x59,0xcf,0xa0]
pm2addu.h s2, t5, a2
# CHECK-ASM-AND-OBJ: pm4addu.b t5, s0, t1
# CHECK-ASM: encoding: [0x3b,0x5f,0x64,0xa4]
pm4addu.b t5, s0, t1
# CHECK-ASM-AND-OBJ: pm2addau.h t3, t1, t5
# CHECK-ASM: encoding: [0x3b,0x5e,0xe3,0xa9]
pm2addau.h t3, t1, t5
# CHECK-ASM-AND-OBJ: pm4addau.b a4, t3, a0
# CHECK-ASM: encoding: [0x3b,0x57,0xae,0xac]
pm4addau.b a4, t3, a0
# CHECK-ASM-AND-OBJ: pmq2add.h t1, a2, t1
# CHECK-ASM: encoding: [0x3b,0x53,0x66,0xb0]
pmq2add.h t1, a2, t1
# CHECK-ASM-AND-OBJ: pmqr2add.h a2, a4, s2
# CHECK-ASM: encoding: [0x3b,0x56,0x27,0xb5]
pmqr2add.h a2, a4, s2
# CHECK-ASM-AND-OBJ: pmq2adda.h a2, s2, t1
# CHECK-ASM: encoding: [0x3b,0x56,0x69,0xb8]
pmq2adda.h a2, s2, t1
# CHECK-ASM-AND-OBJ: pmqr2adda.h t1, s0, a2
# CHECK-ASM: encoding: [0x3b,0x53,0xc4,0xbc]
pmqr2adda.h t1, s0, a2
# CHECK-ASM-AND-OBJ: pm2sub.h t1, a0, a4
# CHECK-ASM: encoding: [0x3b,0x53,0xe5,0xc0]
pm2sub.h t1, a0, a4
# CHECK-ASM-AND-OBJ: pm2sadd.h s2, s2, t3
# CHECK-ASM: encoding: [0x3b,0x59,0xc9,0xc5]
pm2sadd.h s2, s2, t3
# CHECK-ASM-AND-OBJ: pm2suba.h s0, s0, t3
# CHECK-ASM: encoding: [0x3b,0x54,0xc4,0xc9]
pm2suba.h s0, s0, t3
# CHECK-ASM-AND-OBJ: pm2sub.hx a2, a2, a0
# CHECK-ASM: encoding: [0x3b,0x56,0xa6,0xd0]
pm2sub.hx a2, a2, a0
# CHECK-ASM-AND-OBJ: pm2sadd.hx t5, s2, a0
# CHECK-ASM: encoding: [0x3b,0x5f,0xa9,0xd4]
pm2sadd.hx t5, s2, a0
# CHECK-ASM-AND-OBJ: pm2suba.hx a4, a2, t5
# CHECK-ASM: encoding: [0x3b,0x57,0xe6,0xd9]
pm2suba.hx a4, a2, t5
# CHECK-ASM-AND-OBJ: pm2addsu.h s0, s0, s2
# CHECK-ASM: encoding: [0x3b,0x54,0x24,0xe1]
pm2addsu.h s0, s0, s2
# CHECK-ASM-AND-OBJ: pm4addsu.b a0, s0, t3
# CHECK-ASM: encoding: [0x3b,0x55,0xc4,0xe5]
pm4addsu.b a0, s0, t3
# CHECK-ASM-AND-OBJ: pm2addasu.h a4, t1, a2
# CHECK-ASM: encoding: [0x3b,0x57,0xc3,0xe8]
pm2addasu.h a4, t1, a2
# CHECK-ASM-AND-OBJ: pm4addasu.b s2, s0, a2
# CHECK-ASM: encoding: [0x3b,0x59,0xc4,0xec]
pm4addasu.b s2, s0, a2
# CHECK-ASM-AND-OBJ: mqacc.h01 a4, t1, a0
# CHECK-ASM: encoding: [0x3b,0x57,0xa3,0xf8]
mqacc.h01 a4, t1, a0
# CHECK-ASM-AND-OBJ: mqracc.h01 s0, a0, t5
# CHECK-ASM: encoding: [0x3b,0x54,0xe5,0xfd]
mqracc.h01 s0, a0, t5
# CHECK-ASM-AND-OBJ: pas.hx t5, s2, t5
# CHECK-ASM: encoding: [0x3b,0x6f,0xe9,0x81]
pas.hx t5, s2, t5
# CHECK-ASM-AND-OBJ: psa.hx s0, t1, t3
# CHECK-ASM: encoding: [0x3b,0x64,0xc3,0x85]
psa.hx s0, t1, t3
# CHECK-ASM-AND-OBJ: psas.hx t5, s2, a2
# CHECK-ASM: encoding: [0x3b,0x6f,0xc9,0x90]
psas.hx t5, s2, a2
# CHECK-ASM-AND-OBJ: pssa.hx s2, t3, t5
# CHECK-ASM: encoding: [0x3b,0x69,0xee,0x95]
pssa.hx s2, t3, t5
# CHECK-ASM-AND-OBJ: pmseq.h a0, t1, s0
# CHECK-ASM: encoding: [0x3b,0x65,0x83,0xc0]
pmseq.h a0, t1, s0
# CHECK-ASM-AND-OBJ: pmseq.b s0, s2, s0
# CHECK-ASM: encoding: [0x3b,0x64,0x89,0xc4]
pmseq.b s0, s2, s0
# CHECK-ASM-AND-OBJ: pmslt.h t3, a2, a4
# CHECK-ASM: encoding: [0x3b,0x6e,0xe6,0xd0]
pmslt.h t3, a2, a4
# CHECK-ASM-AND-OBJ: pmslt.b t5, a4, t1
# CHECK-ASM: encoding: [0x3b,0x6f,0x67,0xd4]
pmslt.b t5, a4, t1
# CHECK-ASM-AND-OBJ: pmsltu.h s2, s0, s2
# CHECK-ASM: encoding: [0x3b,0x69,0x24,0xd9]
pmsltu.h s2, s0, s2
# CHECK-ASM-AND-OBJ: pmsltu.b s0, s2, t5
# CHECK-ASM: encoding: [0x3b,0x64,0xe9,0xdd]
pmsltu.b s0, s2, t5
# CHECK-ASM-AND-OBJ: pmin.h s2, t3, s2
# CHECK-ASM: encoding: [0x3b,0x69,0x2e,0xe1]
pmin.h s2, t3, s2
# CHECK-ASM-AND-OBJ: pmin.b a2, a2, t5
# CHECK-ASM: encoding: [0x3b,0x66,0xe6,0xe5]
pmin.b a2, a2, t5
# CHECK-ASM-AND-OBJ: pminu.h a0, a4, a2
# CHECK-ASM: encoding: [0x3b,0x65,0xc7,0xe8]
pminu.h a0, a4, a2
# CHECK-ASM-AND-OBJ: pminu.b a4, t3, s2
# CHECK-ASM: encoding: [0x3b,0x67,0x2e,0xed]
pminu.b a4, t3, s2
# CHECK-ASM-AND-OBJ: pmax.h a4, s2, a0
# CHECK-ASM: encoding: [0x3b,0x67,0xa9,0xf0]
pmax.h a4, s2, a0
# CHECK-ASM-AND-OBJ: pmax.b t1, s0, s0
# CHECK-ASM: encoding: [0x3b,0x63,0x84,0xf4]
pmax.b t1, s0, s0
# CHECK-ASM-AND-OBJ: pmaxu.h a0, s0, s0
# CHECK-ASM: encoding: [0x3b,0x65,0x84,0xf8]
pmaxu.h a0, s0, s0
# CHECK-ASM-AND-OBJ: pmaxu.b t3, a0, t1
# CHECK-ASM: encoding: [0x3b,0x6e,0x65,0xfc]
pmaxu.b t3, a0, t1
# CHECK-ASM-AND-OBJ: mseq a4, t1, s0
# CHECK-ASM: encoding: [0x3b,0x67,0x83,0xc2]
mseq a4, t1, s0
# CHECK-ASM-AND-OBJ: mslt t5, t5, t1
# CHECK-ASM: encoding: [0x3b,0x6f,0x6f,0xd2]
mslt t5, t5, t1
# CHECK-ASM-AND-OBJ: msltu s2, a0, t3
# CHECK-ASM: encoding: [0x3b,0x69,0xc5,0xdb]
msltu s2, a0, t3
# CHECK-ASM-AND-OBJ: pmulh.h s0, t3, t3
# CHECK-ASM: encoding: [0x3b,0x74,0xce,0x81]
pmulh.h s0, t3, t3
# CHECK-ASM-AND-OBJ: pmulhr.h t1, t5, s0
# CHECK-ASM: encoding: [0x3b,0x73,0x8f,0x84]
pmulhr.h t1, t5, s0
# CHECK-ASM-AND-OBJ: pmhacc.h s0, t5, s2
# CHECK-ASM: encoding: [0x3b,0x74,0x2f,0x89]
pmhacc.h s0, t5, s2
# CHECK-ASM-AND-OBJ: pmhracc.h a4, t3, a2
# CHECK-ASM: encoding: [0x3b,0x77,0xce,0x8c]
pmhracc.h a4, t3, a2
# CHECK-ASM-AND-OBJ: pmulhu.h a4, t5, a2
# CHECK-ASM: encoding: [0x3b,0x77,0xcf,0x90]
pmulhu.h a4, t5, a2
# CHECK-ASM-AND-OBJ: pmulhru.h a4, a2, a4
# CHECK-ASM: encoding: [0x3b,0x77,0xe6,0x94]
pmulhru.h a4, a2, a4
# CHECK-ASM-AND-OBJ: pmhaccu.h a4, t1, t3
# CHECK-ASM: encoding: [0x3b,0x77,0xc3,0x99]
pmhaccu.h a4, t1, t3
# CHECK-ASM-AND-OBJ: pmhraccu.h s2, s0, t1
# CHECK-ASM: encoding: [0x3b,0x79,0x64,0x9c]
pmhraccu.h s2, s0, t1
# CHECK-ASM-AND-OBJ: pmulh.h.b0 a0, t5, a4
# CHECK-ASM: encoding: [0x3b,0x75,0xef,0xa0]
pmulh.h.b0 a0, t5, a4
# CHECK-ASM-AND-OBJ: pmulhsu.h.b0 s0, a4, s2
# CHECK-ASM: encoding: [0x3b,0x74,0x27,0xa5]
pmulhsu.h.b0 s0, a4, s2
# CHECK-ASM-AND-OBJ: pmhaccu.h.b0 s0, a0, t5
# CHECK-ASM: encoding: [0x3b,0x74,0xe5,0xa9]
pmhaccu.h.b0 s0, a0, t5
# CHECK-ASM-AND-OBJ: pmhaccsu.h.b0 t3, s0, a4
# CHECK-ASM: encoding: [0x3b,0x7e,0xe4,0xac]
pmhaccsu.h.b0 t3, s0, a4
# CHECK-ASM-AND-OBJ: pmulh.h.b1 a0, a4, s2
# CHECK-ASM: encoding: [0x3b,0x75,0x27,0xb1]
pmulh.h.b1 a0, a4, s2
# CHECK-ASM-AND-OBJ: pmulhsu.h.b1 t5, t3, t1
# CHECK-ASM: encoding: [0x3b,0x7f,0x6e,0xb4]
pmulhsu.h.b1 t5, t3, t1
# CHECK-ASM-AND-OBJ: pmhacc.h.b1 t1, t5, a2
# CHECK-ASM: encoding: [0x3b,0x73,0xcf,0xb8]
pmhacc.h.b1 t1, t5, a2
# CHECK-ASM-AND-OBJ: pmhaccsu.h.b1 a2, a0, a4
# CHECK-ASM: encoding: [0x3b,0x76,0xe5,0xbc]
pmhaccsu.h.b1 a2, a0, a4
# CHECK-ASM-AND-OBJ: pmulhsu.h s0, a0, t1
# CHECK-ASM: encoding: [0x3b,0x74,0x65,0xc0]
pmulhsu.h s0, a0, t1
# CHECK-ASM-AND-OBJ: pmulhrsu.h t3, t5, a4
# CHECK-ASM: encoding: [0x3b,0x7e,0xef,0xc4]
pmulhrsu.h t3, t5, a4
# CHECK-ASM-AND-OBJ: pmhaccsu.h s0, s0, a4
# CHECK-ASM: encoding: [0x3b,0x74,0xe4,0xc8]
pmhaccsu.h s0, s0, a4
# CHECK-ASM-AND-OBJ: pmhraccsu.h a2, a2, a0
# CHECK-ASM: encoding: [0x3b,0x76,0xa6,0xcc]
pmhraccsu.h a2, a2, a0
# CHECK-ASM-AND-OBJ: pmulq.h a0, t1, t1
# CHECK-ASM: encoding: [0x3b,0x75,0x63,0xd0]
pmulq.h a0, t1, t1
# CHECK-ASM-AND-OBJ: pmulqr.h s2, s0, s2
# CHECK-ASM: encoding: [0x3b,0x79,0x24,0xd5]
pmulqr.h s2, s0, s2
# CHECK-ASM-AND-OBJ: mulhr a4, s2, t5
# CHECK-ASM: encoding: [0x3b,0x77,0xe9,0x87]
mulhr a4, s2, t5
# CHECK-ASM-AND-OBJ: mhacc t1, s0, a2
# CHECK-ASM: encoding: [0x3b,0x73,0xc4,0x8a]
mhacc t1, s0, a2
# CHECK-ASM-AND-OBJ: mhracc t1, t5, s0
# CHECK-ASM: encoding: [0x3b,0x73,0x8f,0x8e]
mhracc t1, t5, s0
# CHECK-ASM-AND-OBJ: mulhru t1, t5, s0
# CHECK-ASM: encoding: [0x3b,0x73,0x8f,0x96]
mulhru t1, t5, s0
# CHECK-ASM-AND-OBJ: mhaccu t3, a2, s2
# CHECK-ASM: encoding: [0x3b,0x7e,0x26,0x9b]
mhaccu t3, a2, s2
# CHECK-ASM-AND-OBJ: mhraccu a0, t1, a4
# CHECK-ASM: encoding: [0x3b,0x75,0xe3,0x9e]
mhraccu a0, t1, a4
# CHECK-ASM-AND-OBJ: mulh.h0 t3, a4, t1
# CHECK-ASM: encoding: [0x3b,0x7e,0x67,0xa2]
mulh.h0 t3, a4, t1
# CHECK-ASM-AND-OBJ: mulhsu.h0 t1, a2, a0
# CHECK-ASM: encoding: [0x3b,0x73,0xa6,0xa6]
mulhsu.h0 t1, a2, a0
# CHECK-ASM-AND-OBJ: mhacc.h0 s0, a2, t3
# CHECK-ASM: encoding: [0x3b,0x74,0xc6,0xab]
mhacc.h0 s0, a2, t3
# CHECK-ASM-AND-OBJ: mhaccsu.h0 a2, t1, s0
# CHECK-ASM: encoding: [0x3b,0x76,0x83,0xae]
mhaccsu.h0 a2, t1, s0
# CHECK-ASM-AND-OBJ: mulh.h1 t1, t1, t3
# CHECK-ASM: encoding: [0x3b,0x73,0xc3,0xb3]
mulh.h1 t1, t1, t3
# CHECK-ASM-AND-OBJ: mulhsu.h1 t3, a2, t1
# CHECK-ASM: encoding: [0x3b,0x7e,0x66,0xb6]
mulhsu.h1 t3, a2, t1
# CHECK-ASM-AND-OBJ: mhacc.h1 t3, a2, a0
# CHECK-ASM: encoding: [0x3b,0x7e,0xa6,0xba]
mhacc.h1 t3, a2, a0
# CHECK-ASM-AND-OBJ: mhaccsu.h1 s0, s2, s0
# CHECK-ASM: encoding: [0x3b,0x74,0x89,0xbe]
mhaccsu.h1 s0, s2, s0
# CHECK-ASM-AND-OBJ: mulhrsu.h t5, a4, t5
# CHECK-ASM: encoding: [0x3b,0x7f,0xe7,0xc7]
mulhrsu.h t5, a4, t5
# CHECK-ASM-AND-OBJ: mhaccsu a0, a2, s2
# CHECK-ASM: encoding: [0x3b,0x75,0x26,0xcb]
mhaccsu a0, a2, s2
# CHECK-ASM-AND-OBJ: mhraccsu a0, a0, t1
# CHECK-ASM: encoding: [0x3b,0x75,0x65,0xce]
mhraccsu a0, a0, t1
# CHECK-ASM-AND-OBJ: mulq t1, a2, a2
# CHECK-ASM: encoding: [0x3b,0x73,0xc6,0xd2]
mulq t1, a2, a2
# CHECK-ASM-AND-OBJ: mulqr a4, a4, t3
# CHECK-ASM: encoding: [0x3b,0x77,0xc7,0xd7]
mulqr a4, a4, t3
# CHECK-ASM-AND-OBJ: mqacc.h00 a2, t3, t3
# CHECK-ASM: encoding: [0x3b,0x76,0xce,0xe9]
mqacc.h00 a2, t3, t3
# CHECK-ASM-AND-OBJ: mqracc.h00 t5, a4, t3
# CHECK-ASM: encoding: [0x3b,0x7f,0xc7,0xed]
mqracc.h00 t5, a4, t3
# CHECK-ASM-AND-OBJ: mqacc.h11 t5, t5, s0
# CHECK-ASM: encoding: [0x3b,0x7f,0x8f,0xf8]
mqacc.h11 t5, t5, s0
# CHECK-ASM-AND-OBJ: mqracc.h11 s0, t5, s2
# CHECK-ASM: encoding: [0x3b,0x74,0x2f,0xfd]
mqracc.h11 s0, t5, s2
# CHECK-ASM-AND-OBJ: pwslli.b a0, t1
# CHECK-ASM: encoding: [0x1b,0x25,0x03,0x01]
pwslli.b a0, t1
# CHECK-ASM-AND-OBJ: pwslli.h s0, a0
# CHECK-ASM: encoding: [0x1b,0x24,0x05,0x12]
pwslli.h s0, a0
# CHECK-ASM-AND-OBJ: wslli s2, t3
# CHECK-ASM: encoding: [0x1b,0x29,0x0e,0x24]
wslli s2, t3
# CHECK-ASM-AND-OBJ: pwslai.b t5, t5
# CHECK-ASM: encoding: [0x1b,0x2f,0x0f,0x41]
pwslai.b t5, t5
# CHECK-ASM-AND-OBJ: pwslai.h t5, a4
# CHECK-ASM: encoding: [0x1b,0x2f,0x07,0x42]
pwslai.h t5, a4
# CHECK-ASM-AND-OBJ: wslai t1, a2
# CHECK-ASM: encoding: [0x1b,0x23,0x06,0x44]
wslai t1, a2
# CHECK-ASM-AND-OBJ: pli.dh s0, 32
# CHECK-ASM: encoding: [0x1b,0x24,0x10,0x30]
pli.dh s0, 32
# CHECK-ASM-AND-OBJ: pli.db a2, 1
# CHECK-ASM: encoding: [0x1b,0x26,0x01,0x34]
pli.db a2, 1
# CHECK-ASM-AND-OBJ: plui.dh t5, 16
# CHECK-ASM: encoding: [0x1b,0x2f,0x08,0x70]
plui.dh t5, 16
# CHECK-ASM-AND-OBJ: pwslli.bs t3, t1, s0
# CHECK-ASM: encoding: [0x1b,0x2e,0x83,0x08]
pwslli.bs t3, t1, s0
# CHECK-ASM-AND-OBJ: pwsll.hs s0, a4, t1
# CHECK-ASM: encoding: [0x1b,0x24,0x67,0x0a]
pwsll.hs s0, a4, t1
# CHECK-ASM-AND-OBJ: wsll a0, s0, s2
# CHECK-ASM: encoding: [0x1b,0x25,0x24,0x0f]
wsll a0, s0, s2
# CHECK-ASM-AND-OBJ: pwsla.bs s0, s0, s0
# CHECK-ASM: encoding: [0x1b,0x24,0x84,0x48]
pwsla.bs s0, s0, s0
# CHECK-ASM-AND-OBJ: pwsla.hs a4, a2, t5
# CHECK-ASM: encoding: [0x1b,0x27,0xe6,0x4b]
pwsla.hs a4, a2, t5
# CHECK-ASM-AND-OBJ: wsla s0, a0, s2
# CHECK-ASM: encoding: [0x1b,0x24,0x25,0x4f]
wsla s0, a0, s2
# CHECK-ASM-AND-OBJ: wzip8p t1, s2, a2
# CHECK-ASM: encoding: [0x1b,0x23,0xc9,0x78]
wzip8p t1, s2, a2
# CHECK-ASM-AND-OBJ: wzip16p s2, t3, s2
# CHECK-ASM: encoding: [0x1b,0x29,0x2e,0x7b]
wzip16p s2, t3, s2
# CHECK-ASM-AND-OBJ: pwadd.h a4, a2, a0
# CHECK-ASM: encoding: [0xbb,0x27,0x56,0x00]
pwadd.h a4, a2, a0
# CHECK-ASM-AND-OBJ: wadd t1, t5, t5
# CHECK-ASM: encoding: [0xbb,0x23,0xff,0x02]
wadd t1, t5, t5
# CHECK-ASM-AND-OBJ: pwadd.b s0, t3, a4
# CHECK-ASM: encoding: [0xbb,0x24,0x7e,0x04]
pwadd.b s0, t3, a4
# CHECK-ASM-AND-OBJ: pw2wadd.h t3, t3, a4
# CHECK-ASM: encoding: [0xbb,0x2e,0x7e,0x06]
pw2wadd.h t3, t3, a4
# CHECK-ASM-AND-OBJ: pwadda.h t1, t3, s2
# CHECK-ASM: encoding: [0xbb,0x23,0x9e,0x08]
pwadda.h t1, t3, s2
# CHECK-ASM-AND-OBJ: wadda s2, t1, a0
# CHECK-ASM: encoding: [0xbb,0x29,0x53,0x0a]
wadda s2, t1, a0
# CHECK-ASM-AND-OBJ: pwadda.b a2, a2, s2
# CHECK-ASM: encoding: [0xbb,0x26,0x96,0x0c]
pwadda.b a2, a2, s2
# CHECK-ASM-AND-OBJ: pw2wadda.h t3, t1, t1
# CHECK-ASM: encoding: [0xbb,0x2e,0x33,0x0e]
pw2wadda.h t3, t1, t1
# CHECK-ASM-AND-OBJ: pwaddu.h a2, t1, a4
# CHECK-ASM: encoding: [0xbb,0x26,0x73,0x10]
pwaddu.h a2, t1, a4
# CHECK-ASM-AND-OBJ: waddu t1, s0, t1
# CHECK-ASM: encoding: [0xbb,0x23,0x34,0x12]
waddu t1, s0, t1
# CHECK-ASM-AND-OBJ: pwaddu.b s2, a0, t3
# CHECK-ASM: encoding: [0xbb,0x29,0xe5,0x14]
pwaddu.b s2, a0, t3
# CHECK-ASM-AND-OBJ: pw2wadd.hx a0, s0, t1
# CHECK-ASM: encoding: [0xbb,0x25,0x34,0x16]
pw2wadd.hx a0, s0, t1
# CHECK-ASM-AND-OBJ: pwaddau.h t1, t5, t3
# CHECK-ASM: encoding: [0xbb,0x23,0xef,0x18]
pwaddau.h t1, t5, t3
# CHECK-ASM-AND-OBJ: waddau t3, s0, s0
# CHECK-ASM: encoding: [0xbb,0x2e,0x44,0x1a]
waddau t3, s0, s0
# CHECK-ASM-AND-OBJ: pwaddau.b a0, a0, t1
# CHECK-ASM: encoding: [0xbb,0x25,0x35,0x1c]
pwaddau.b a0, a0, t1
# CHECK-ASM-AND-OBJ: pw2wadda.hx a4, a2, t3
# CHECK-ASM: encoding: [0xbb,0x27,0xe6,0x1e]
pw2wadda.hx a4, a2, t3
# CHECK-ASM-AND-OBJ: pwmul.h s2, t1, a0
# CHECK-ASM: encoding: [0xbb,0x29,0x53,0x20]
pwmul.h s2, t1, a0
# CHECK-ASM-AND-OBJ: wmul t3, a2, t5
# CHECK-ASM: encoding: [0xbb,0x2e,0xf6,0x22]
wmul t3, a2, t5
# CHECK-ASM-AND-OBJ: pwmul.b a0, s0, s2
# CHECK-ASM: encoding: [0xbb,0x25,0x94,0x24]
pwmul.b a0, s0, s2
# CHECK-ASM-AND-OBJ: pw2waddu.h s2, s2, a0
# CHECK-ASM: encoding: [0xbb,0x29,0x59,0x26]
pw2waddu.h s2, s2, a0
# CHECK-ASM-AND-OBJ: pwmacc.h s0, a2, a2
# CHECK-ASM: encoding: [0xbb,0x24,0x66,0x28]
pwmacc.h s0, a2, a2
# CHECK-ASM-AND-OBJ: wmacc a0, a2, a2
# CHECK-ASM: encoding: [0xbb,0x25,0x66,0x2a]
wmacc a0, a2, a2
# CHECK-ASM-AND-OBJ: pm2waddau.h t5, a2, t5
# CHECK-ASM: encoding: [0xbb,0x2f,0xf6,0x2e]
pm2waddau.h t5, a2, t5
# CHECK-ASM-AND-OBJ: pwmulu.h a2, a0, t5
# CHECK-ASM: encoding: [0xbb,0x26,0xf5,0x30]
pwmulu.h a2, a0, t5
# CHECK-ASM-AND-OBJ: wmulu s2, a0, t3
# CHECK-ASM: encoding: [0xbb,0x29,0xe5,0x32]
wmulu s2, a0, t3
# CHECK-ASM-AND-OBJ: pwmulu.b a2, a4, a2
# CHECK-ASM: encoding: [0xbb,0x26,0x67,0x34]
pwmulu.b a2, a4, a2
# CHECK-ASM-AND-OBJ: pwmaccu.h t1, a4, a2
# CHECK-ASM: encoding: [0xbb,0x23,0x67,0x38]
pwmaccu.h t1, a4, a2
# CHECK-ASM-AND-OBJ: wmaccu a2, a0, t1
# CHECK-ASM: encoding: [0xbb,0x26,0x35,0x3a]
wmaccu a2, a0, t1
# CHECK-ASM-AND-OBJ: pwsub.h s0, s2, t3
# CHECK-ASM: encoding: [0xbb,0x24,0xe9,0x40]
pwsub.h s0, s2, t3
# CHECK-ASM-AND-OBJ: wsub t1, s2, a4
# CHECK-ASM: encoding: [0xbb,0x23,0x79,0x42]
wsub t1, s2, a4
# CHECK-ASM-AND-OBJ: pwsub.b a4, s2, s2
# CHECK-ASM: encoding: [0xbb,0x27,0x99,0x44]
pwsub.b a4, s2, s2
# CHECK-ASM-AND-OBJ: pw2wsub.h t1, a4, t3
# CHECK-ASM: encoding: [0xbb,0x23,0xe7,0x46]
pw2wsub.h t1, a4, t3
# CHECK-ASM-AND-OBJ: pwsuba.h a4, t5, t1
# CHECK-ASM: encoding: [0xbb,0x27,0x3f,0x48]
pwsuba.h a4, t5, t1
# CHECK-ASM-AND-OBJ: wsuba a0, s0, t5
# CHECK-ASM: encoding: [0xbb,0x25,0xf4,0x4a]
wsuba a0, s0, t5
# CHECK-ASM-AND-OBJ: pwsuba.b a0, a2, s2
# CHECK-ASM: encoding: [0xbb,0x25,0x96,0x4c]
pwsuba.b a0, a2, s2
# CHECK-ASM-AND-OBJ: pw2wsuba.h t5, s0, s2
# CHECK-ASM: encoding: [0xbb,0x2f,0x94,0x4e]
pw2wsuba.h t5, s0, s2
# CHECK-ASM-AND-OBJ: pwsubu.h t5, s2, a2
# CHECK-ASM: encoding: [0xbb,0x2f,0x69,0x50]
pwsubu.h t5, s2, a2
# CHECK-ASM-AND-OBJ: wsubu a2, a4, a0
# CHECK-ASM: encoding: [0xbb,0x26,0x57,0x52]
wsubu a2, a4, a0
# CHECK-ASM-AND-OBJ: pwsubu.b a2, a2, t5
# CHECK-ASM: encoding: [0xbb,0x26,0xf6,0x54]
pwsubu.b a2, a2, t5
# CHECK-ASM-AND-OBJ: pw2wsub.hx t5, a2, a0
# CHECK-ASM: encoding: [0xbb,0x2f,0x56,0x56]
pw2wsub.hx t5, a2, a0
# CHECK-ASM-AND-OBJ: pwsubau.h t5, s0, s2
# CHECK-ASM: encoding: [0xbb,0x2f,0x94,0x58]
pwsubau.h t5, s0, s2
# CHECK-ASM-AND-OBJ: wsubau t5, a0, t1
# CHECK-ASM: encoding: [0xbb,0x2f,0x35,0x5a]
wsubau t5, a0, t1
# CHECK-ASM-AND-OBJ: pwsubau.b a2, a0, a0
# CHECK-ASM: encoding: [0xbb,0x26,0x55,0x5c]
pwsubau.b a2, a0, a0
# CHECK-ASM-AND-OBJ: pw2wsuba.hx a2, a2, t5
# CHECK-ASM: encoding: [0xbb,0x26,0xf6,0x5e]
pw2wsuba.hx a2, a2, t5
# CHECK-ASM-AND-OBJ: pwmulsu.h s0, a2, t3
# CHECK-ASM: encoding: [0xbb,0x24,0xe6,0x60]
pwmulsu.h s0, a2, t3
# CHECK-ASM-AND-OBJ: wmulsu a0, s2, a0
# CHECK-ASM: encoding: [0xbb,0x25,0x59,0x62]
wmulsu a0, s2, a0
# CHECK-ASM-AND-OBJ: pwmulsu.b t3, t5, t1
# CHECK-ASM: encoding: [0xbb,0x2e,0x3f,0x64]
pwmulsu.b t3, t5, t1
# CHECK-ASM-AND-OBJ: pm2waddsu.h a4, a2, s2
# CHECK-ASM: encoding: [0xbb,0x27,0x96,0x66]
pm2waddsu.h a4, a2, s2
# CHECK-ASM-AND-OBJ: pwmaccsu.h t5, s2, a2
# CHECK-ASM: encoding: [0xbb,0x2f,0x69,0x68]
pwmaccsu.h t5, s2, a2
# CHECK-ASM-AND-OBJ: wmaccsu t3, s0, a4
# CHECK-ASM: encoding: [0xbb,0x2e,0x74,0x6a]
wmaccsu t3, s0, a4
# CHECK-ASM-AND-OBJ: pm2waddasu.h t3, t5, a0
# CHECK-ASM: encoding: [0xbb,0x2e,0x5f,0x6e]
pm2waddasu.h t3, t5, a0
# CHECK-ASM-AND-OBJ: pmqwacc.h t5, t5, a2
# CHECK-ASM: encoding: [0xbb,0x2f,0x6f,0x78]
pmqwacc.h t5, t5, a2
# CHECK-ASM-AND-OBJ: pmqwacc s2, a4, a2
# CHECK-ASM: encoding: [0xbb,0x29,0x67,0x7a]
pmqwacc s2, a4, a2
# CHECK-ASM-AND-OBJ: pmqrwacc.h a4, t3, a4
# CHECK-ASM: encoding: [0xbb,0x27,0x7e,0x7c]
pmqrwacc.h a4, t3, a4
# CHECK-ASM-AND-OBJ: pmqrwacc s0, s2, t5
# CHECK-ASM: encoding: [0xbb,0x24,0xf9,0x7e]
pmqrwacc s0, s2, t5
# CHECK-ASM-AND-OBJ: predsum.dhs s0, t3, a0
# CHECK-ASM: encoding: [0x1b,0x44,0xae,0x08]
predsum.dhs s0, t3, a0
# CHECK-ASM-AND-OBJ: predsum.dbs a2, s0, t3
# CHECK-ASM: encoding: [0x1b,0x46,0xc4,0x0d]
predsum.dbs a2, s0, t3
# CHECK-ASM-AND-OBJ: predsumu.dhs a2, a2, t3
# CHECK-ASM: encoding: [0x1b,0x46,0xc6,0x39]
predsumu.dhs a2, a2, t3
# CHECK-ASM-AND-OBJ: predsumu.dbs a2, a0, s0
# CHECK-ASM: encoding: [0x1b,0x46,0x85,0x3c]
predsumu.dbs a2, a0, s0
# CHECK-ASM-AND-OBJ: pnsrli.b a2, a0
# CHECK-ASM: encoding: [0x1b,0xc6,0x05,0x01]
pnsrli.b a2, a0
# CHECK-ASM-AND-OBJ: pnsrli.h a0, t3
# CHECK-ASM: encoding: [0x1b,0xc5,0x0e,0x02]
pnsrli.h a0, t3
# CHECK-ASM-AND-OBJ: nsrli a2, a0
# CHECK-ASM: encoding: [0x1b,0xc6,0x05,0x04]
nsrli a2, a0
# CHECK-ASM-AND-OBJ: pnclipiu.b a4, t3
# CHECK-ASM: encoding: [0x1b,0xc7,0x0e,0x21]
pnclipiu.b a4, t3
# CHECK-ASM-AND-OBJ: pnclipiu.h t1, s2
# CHECK-ASM: encoding: [0x1b,0xc3,0x09,0x22]
pnclipiu.h t1, s2
# CHECK-ASM-AND-OBJ: nclipiu s0, a2
# CHECK-ASM: encoding: [0x1b,0xc4,0x06,0x24]
nclipiu s0, a2
# CHECK-ASM-AND-OBJ: pnclipriu.b s2, s0
# CHECK-ASM: encoding: [0x1b,0xc9,0x04,0x31]
pnclipriu.b s2, s0
# CHECK-ASM-AND-OBJ: pnclipriu.h s0, s0
# CHECK-ASM: encoding: [0x1b,0xc4,0x04,0x32]
pnclipriu.h s0, s0
# CHECK-ASM-AND-OBJ: nclipriu t3, t3
# CHECK-ASM: encoding: [0x1b,0xce,0x0e,0x34]
nclipriu t3, t3
# CHECK-ASM-AND-OBJ: pnsrai.b s2, t5
# CHECK-ASM: encoding: [0x1b,0xc9,0x0f,0x41]
pnsrai.b s2, t5
# CHECK-ASM-AND-OBJ: pnsrai.h s0, a0
# CHECK-ASM: encoding: [0x1b,0xc4,0x05,0x42]
pnsrai.h s0, a0
# CHECK-ASM-AND-OBJ: nsrai a4, t3
# CHECK-ASM: encoding: [0x1b,0xc7,0x0e,0x44]
nsrai a4, t3
# CHECK-ASM-AND-OBJ: pnsari.b t5, t5
# CHECK-ASM: encoding: [0x1b,0xcf,0x0f,0x51]
pnsari.b t5, t5
# CHECK-ASM-AND-OBJ: pnsari.h t1, a4
# CHECK-ASM: encoding: [0x1b,0xc3,0x07,0x52]
pnsari.h t1, a4
# CHECK-ASM-AND-OBJ: nsari s0, t1
# CHECK-ASM: encoding: [0x1b,0xc4,0x03,0x54]
nsari s0, t1
# CHECK-ASM-AND-OBJ: pnclipi.b t1, a4
# CHECK-ASM: encoding: [0x1b,0xc3,0x07,0x61]
pnclipi.b t1, a4
# CHECK-ASM-AND-OBJ: pnclipi.h a0, a4
# CHECK-ASM: encoding: [0x1b,0xc5,0x07,0x62]
pnclipi.h a0, a4
# CHECK-ASM-AND-OBJ: nclipi t5, t5
# CHECK-ASM: encoding: [0x1b,0xcf,0x0f,0x64]
nclipi t5, t5
# CHECK-ASM-AND-OBJ: pnclipri.b a0, s0
# CHECK-ASM: encoding: [0x1b,0xc5,0x04,0x71]
pnclipri.b a0, s0
# CHECK-ASM-AND-OBJ: pnclipri.h s2, t5
# CHECK-ASM: encoding: [0x1b,0xc9,0x0f,0x72]
pnclipri.h s2, t5
# CHECK-ASM-AND-OBJ: nclipri t3, s0
# CHECK-ASM: encoding: [0x1b,0xce,0x04,0x74]
nclipri t3, s0
# CHECK-ASM-AND-OBJ: pnsrl.bs t3, s0, a4
# CHECK-ASM: encoding: [0x1b,0xce,0xe4,0x08]
pnsrl.bs t3, s0, a4
# CHECK-ASM-AND-OBJ: pnsrl.hs a2, t1, a4
# CHECK-ASM: encoding: [0x1b,0xc6,0xe3,0x0a]
pnsrl.hs a2, t1, a4
# CHECK-ASM-AND-OBJ: nsrl a2, a2, a0
# CHECK-ASM: encoding: [0x1b,0xc6,0xa6,0x0e]
nsrl a2, a2, a0
# CHECK-ASM-AND-OBJ: pnclipu.bs a4, t5, a2
# CHECK-ASM: encoding: [0x1b,0xc7,0xcf,0x28]
pnclipu.bs a4, t5, a2
# CHECK-ASM-AND-OBJ: pnclipu.hs t1, a2, a4
# CHECK-ASM: encoding: [0x1b,0xc3,0xe6,0x2a]
pnclipu.hs t1, a2, a4
# CHECK-ASM-AND-OBJ: nclipu t5, s2, t1
# CHECK-ASM: encoding: [0x1b,0xcf,0x69,0x2e]
nclipu t5, s2, t1
# CHECK-ASM-AND-OBJ: pnclipru.bs t5, s2, s2
# CHECK-ASM: encoding: [0x1b,0xcf,0x29,0x39]
pnclipru.bs t5, s2, s2
# CHECK-ASM-AND-OBJ: pnclipru.hs t5, s2, a0
# CHECK-ASM: encoding: [0x1b,0xcf,0xa9,0x3a]
pnclipru.hs t5, s2, a0
# CHECK-ASM-AND-OBJ: nclipru a4, t5, t5
# CHECK-ASM: encoding: [0x1b,0xc7,0xef,0x3f]
nclipru a4, t5, t5
# CHECK-ASM-AND-OBJ: pnsra.bs a4, t1, a4
# CHECK-ASM: encoding: [0x1b,0xc7,0xe3,0x48]
pnsra.bs a4, t1, a4
# CHECK-ASM-AND-OBJ: pnsra.hs s0, s2, t3
# CHECK-ASM: encoding: [0x1b,0xc4,0xc9,0x4b]
pnsra.hs s0, s2, t3
# CHECK-ASM-AND-OBJ: nsra t1, s0, a4
# CHECK-ASM: encoding: [0x1b,0xc3,0xe4,0x4e]
nsra t1, s0, a4
# CHECK-ASM-AND-OBJ: pnsrar.bs a2, s0, a4
# CHECK-ASM: encoding: [0x1b,0xc6,0xe4,0x58]
pnsrar.bs a2, s0, a4
# CHECK-ASM-AND-OBJ: pnsrar.hs s0, a4, a0
# CHECK-ASM: encoding: [0x1b,0xc4,0xa7,0x5a]
pnsrar.hs s0, a4, a0
# CHECK-ASM-AND-OBJ: nsrar a4, a4, s0
# CHECK-ASM: encoding: [0x1b,0xc7,0x87,0x5e]
nsrar a4, a4, s0
# CHECK-ASM-AND-OBJ: pnclip.bs t1, t5, t3
# CHECK-ASM: encoding: [0x1b,0xc3,0xcf,0x69]
pnclip.bs t1, t5, t3
# CHECK-ASM-AND-OBJ: pnclip.hs a0, a2, a0
# CHECK-ASM: encoding: [0x1b,0xc5,0xa6,0x6a]
pnclip.hs a0, a2, a0
# CHECK-ASM-AND-OBJ: nclip t3, t5, t3
# CHECK-ASM: encoding: [0x1b,0xce,0xcf,0x6f]
nclip t3, t5, t3
# CHECK-ASM-AND-OBJ: pnclipr.bs t1, a2, a0
# CHECK-ASM: encoding: [0x1b,0xc3,0xa6,0x78]
pnclipr.bs t1, a2, a0
# CHECK-ASM-AND-OBJ: pnclipr.hs a4, s2, t3
# CHECK-ASM: encoding: [0x1b,0xc7,0xc9,0x7b]
pnclipr.hs a4, s2, t3
# CHECK-ASM-AND-OBJ: nclipr t1, t5, a2
# CHECK-ASM: encoding: [0x1b,0xc3,0xcf,0x7e]
nclipr t1, t5, a2
# CHECK-ASM-AND-OBJ: pslli.db a0, s2
# CHECK-ASM: encoding: [0x1b,0x65,0x89,0x00]
pslli.db a0, s2
# CHECK-ASM-AND-OBJ: pslli.dh t3, t1
# CHECK-ASM: encoding: [0x1b,0x6e,0x03,0x01]
pslli.dh t3, t1
# CHECK-ASM-AND-OBJ: pslli.dw a4, t3
# CHECK-ASM: encoding: [0x1b,0x67,0x0e,0x02]
pslli.dw a4, t3
# CHECK-ASM-AND-OBJ: psslai.dh t1, a4
# CHECK-ASM: encoding: [0x1b,0x63,0x07,0x51]
psslai.dh t1, a4
# CHECK-ASM-AND-OBJ: psslai.dw a0, t3
# CHECK-ASM: encoding: [0x1b,0x65,0x0e,0x52]
psslai.dw a0, t3
# CHECK-ASM-AND-OBJ: psext.dh.b t1, t5
# CHECK-ASM: encoding: [0x1b,0x23,0x4f,0x60]
psext.dh.b t1, t5
# CHECK-ASM-AND-OBJ: psext.dw.b t5, t5
# CHECK-ASM: encoding: [0x1b,0x2f,0x4f,0x62]
psext.dw.b t5, t5
# CHECK-ASM-AND-OBJ: psext.dw.h s0, t1
# CHECK-ASM: encoding: [0x1b,0x24,0x53,0x62]
psext.dw.h s0, t1
# CHECK-ASM-AND-OBJ: psabs.dh s0, s2
# CHECK-ASM: encoding: [0x1b,0x24,0x79,0x60]
psabs.dh s0, s2
# CHECK-ASM-AND-OBJ: psabs.db s2, a2
# CHECK-ASM: encoding: [0x1b,0x29,0x76,0x64]
psabs.db s2, a2
# CHECK-ASM-AND-OBJ: psll.dhs s2, t3, a4
# CHECK-ASM: encoding: [0x1b,0x69,0xee,0x08]
psll.dhs s2, t3, a4
# CHECK-ASM-AND-OBJ: psll.dws a2, t1, t3
# CHECK-ASM: encoding: [0x1b,0x66,0xc3,0x0b]
psll.dws a2, t1, t3
# CHECK-ASM-AND-OBJ: psll.dbs a0, a4, a2
# CHECK-ASM: encoding: [0x1b,0x65,0xc7,0x0c]
psll.dbs a0, a4, a2
# CHECK-ASM-AND-OBJ: padd.dhs t1, a4, s2
# CHECK-ASM: encoding: [0x1b,0x63,0x27,0x19]
padd.dhs t1, a4, s2
# CHECK-ASM-AND-OBJ: padd.dws a4, a4, t3
# CHECK-ASM: encoding: [0x1b,0x67,0xc7,0x1b]
padd.dws a4, a4, t3
# CHECK-ASM-AND-OBJ: padd.dbs a2, a4, t3
# CHECK-ASM: encoding: [0x1b,0x66,0xc7,0x1d]
padd.dbs a2, a4, t3
# CHECK-ASM-AND-OBJ: pssha.dhs a0, s0, s2
# CHECK-ASM: encoding: [0x1b,0x65,0x24,0x69]
pssha.dhs a0, s0, s2
# CHECK-ASM-AND-OBJ: pssha.dws a0, t1, s2
# CHECK-ASM: encoding: [0x1b,0x65,0x23,0x6b]
pssha.dws a0, t1, s2
# CHECK-ASM-AND-OBJ: psshar.dhs a2, a4, t3
# CHECK-ASM: encoding: [0x1b,0x66,0xc7,0x79]
psshar.dhs a2, a4, t3
# CHECK-ASM-AND-OBJ: psshar.dws s0, t3, s0
# CHECK-ASM: encoding: [0x1b,0x64,0x8e,0x7a]
psshar.dws s0, t3, s0
# CHECK-ASM-AND-OBJ: psrli.db t5, a2
# CHECK-ASM: encoding: [0x1b,0xef,0x86,0x00]
psrli.db t5, a2
# CHECK-ASM-AND-OBJ: psrli.dh a2, t3
# CHECK-ASM: encoding: [0x1b,0xe6,0x0e,0x01]
psrli.dh a2, t3
# CHECK-ASM-AND-OBJ: psrli.dw s2, t1
# CHECK-ASM: encoding: [0x1b,0xe9,0x03,0x02]
psrli.dw s2, t1
# CHECK-ASM-AND-OBJ: pusati.dh a0, a4
# CHECK-ASM: encoding: [0x1b,0xe5,0x07,0x21]
pusati.dh a0, a4
# CHECK-ASM-AND-OBJ: pusati.dw a0, s2
# CHECK-ASM: encoding: [0x1b,0xe5,0x09,0x22]
pusati.dw a0, s2
# CHECK-ASM-AND-OBJ: psrai.db t5, t5
# CHECK-ASM: encoding: [0x1b,0xef,0x8f,0x40]
psrai.db t5, t5
# CHECK-ASM-AND-OBJ: psrai.dh s0, a2
# CHECK-ASM: encoding: [0x1b,0xe4,0x06,0x41]
psrai.dh s0, a2
# CHECK-ASM-AND-OBJ: psrai.dw t5, a0
# CHECK-ASM: encoding: [0x1b,0xef,0x05,0x42]
psrai.dw t5, a0
# CHECK-ASM-AND-OBJ: psrari.dh a2, a2
# CHECK-ASM: encoding: [0x1b,0xe6,0x06,0x51]
psrari.dh a2, a2
# CHECK-ASM-AND-OBJ: psrari.dw a4, a0
# CHECK-ASM: encoding: [0x1b,0xe7,0x05,0x52]
psrari.dw a4, a0
# CHECK-ASM-AND-OBJ: psati.dh s2, s2
# CHECK-ASM: encoding: [0x1b,0xe9,0x09,0x61]
psati.dh s2, s2
# CHECK-ASM-AND-OBJ: psati.dw t5, t3
# CHECK-ASM: encoding: [0x1b,0xef,0x0e,0x62]
psati.dw t5, t3
# CHECK-ASM-AND-OBJ: psrl.dhs a0, t1, t5
# CHECK-ASM: encoding: [0x1b,0xe5,0xe3,0x09]
psrl.dhs a0, t1, t5
# CHECK-ASM-AND-OBJ: psrl.dws s0, s2, t1
# CHECK-ASM: encoding: [0x1b,0xe4,0x69,0x0a]
psrl.dws s0, s2, t1
# CHECK-ASM-AND-OBJ: psrl.dbs a0, s0, t5
# CHECK-ASM: encoding: [0x1b,0xe5,0xe4,0x0d]
psrl.dbs a0, s0, t5
# CHECK-ASM-AND-OBJ: psra.dhs a4, t3, t1
# CHECK-ASM: encoding: [0x1b,0xe7,0x6e,0x48]
psra.dhs a4, t3, t1
# CHECK-ASM-AND-OBJ: psra.dws a2, s2, t1
# CHECK-ASM: encoding: [0x1b,0xe6,0x69,0x4a]
psra.dws a2, s2, t1
# CHECK-ASM-AND-OBJ: psra.dbs s0, t1, t5
# CHECK-ASM: encoding: [0x1b,0xe4,0xe3,0x4d]
psra.dbs s0, t1, t5
# CHECK-ASM-AND-OBJ: padd.dh s2, a4, a2
# CHECK-ASM: encoding: [0x1b,0x69,0xc7,0x80]
padd.dh s2, a4, a2
# CHECK-ASM-AND-OBJ: padd.dw a2, s2, a2
# CHECK-ASM: encoding: [0x1b,0x66,0xc9,0x82]
padd.dw a2, s2, a2
# CHECK-ASM-AND-OBJ: padd.db a4, a2, a2
# CHECK-ASM: encoding: [0x1b,0x67,0xc6,0x84]
padd.db a4, a2, a2
# CHECK-ASM-AND-OBJ: addd t1, s2, s0
# CHECK-ASM: encoding: [0x1b,0x63,0x89,0x86]
addd t1, s2, s0
# CHECK-ASM-AND-OBJ: psadd.dh t3, s2, t3
# CHECK-ASM: encoding: [0x1b,0x6e,0xc9,0x91]
psadd.dh t3, s2, t3
# CHECK-ASM-AND-OBJ: psadd.dw a4, t3, t3
# CHECK-ASM: encoding: [0x1b,0x67,0xce,0x93]
psadd.dw a4, t3, t3
# CHECK-ASM-AND-OBJ: psadd.db t5, s0, a2
# CHECK-ASM: encoding: [0x1b,0x6f,0xc4,0x94]
psadd.db t5, s0, a2
# CHECK-ASM-AND-OBJ: paadd.dh t1, s2, a0
# CHECK-ASM: encoding: [0x1b,0x63,0xa9,0x98]
paadd.dh t1, s2, a0
# CHECK-ASM-AND-OBJ: paadd.dw a4, a2, s0
# CHECK-ASM: encoding: [0x1b,0x67,0x86,0x9a]
paadd.dw a4, a2, s0
# CHECK-ASM-AND-OBJ: paadd.db t5, t3, s0
# CHECK-ASM: encoding: [0x1b,0x6f,0x8e,0x9c]
paadd.db t5, t3, s0
# CHECK-ASM-AND-OBJ: psaddu.dh a4, a2, t5
# CHECK-ASM: encoding: [0x1b,0x67,0xe6,0xb1]
psaddu.dh a4, a2, t5
# CHECK-ASM-AND-OBJ: psaddu.dw a4, t5, s2
# CHECK-ASM: encoding: [0x1b,0x67,0x2f,0xb3]
psaddu.dw a4, t5, s2
# CHECK-ASM-AND-OBJ: psaddu.db a4, a0, t1
# CHECK-ASM: encoding: [0x1b,0x67,0x65,0xb4]
psaddu.db a4, a0, t1
# CHECK-ASM-AND-OBJ: paaddu.dh a4, a4, s2
# CHECK-ASM: encoding: [0x1b,0x67,0x27,0xb9]
paaddu.dh a4, a4, s2
# CHECK-ASM-AND-OBJ: paaddu.dw t3, s0, t5
# CHECK-ASM: encoding: [0x1b,0x6e,0xe4,0xbb]
paaddu.dw t3, s0, t5
# CHECK-ASM-AND-OBJ: paaddu.db a0, s0, s0
# CHECK-ASM: encoding: [0x1b,0x65,0x84,0xbc]
paaddu.db a0, s0, s0
# CHECK-ASM-AND-OBJ: psub.dh t5, a4, a4
# CHECK-ASM: encoding: [0x1b,0x6f,0xe7,0xc0]
psub.dh t5, a4, a4
# CHECK-ASM-AND-OBJ: psub.dw t1, s0, t5
# CHECK-ASM: encoding: [0x1b,0x63,0xe4,0xc3]
psub.dw t1, s0, t5
# CHECK-ASM-AND-OBJ: psub.db a4, a0, t5
# CHECK-ASM: encoding: [0x1b,0x67,0xe5,0xc5]
psub.db a4, a0, t5
# CHECK-ASM-AND-OBJ: subd a2, a4, t1
# CHECK-ASM: encoding: [0x1b,0x66,0x67,0xc6]
subd a2, a4, t1
# CHECK-ASM-AND-OBJ: pdif.dh t5, t1, t3
# CHECK-ASM: encoding: [0x1b,0x6f,0xc3,0xc9]
pdif.dh t5, t1, t3
# CHECK-ASM-AND-OBJ: pdif.db t1, t5, a0
# CHECK-ASM: encoding: [0x1b,0x63,0xaf,0xcc]
pdif.db t1, t5, a0
# CHECK-ASM-AND-OBJ: pssub.dh s0, s2, s2
# CHECK-ASM: encoding: [0x1b,0x64,0x29,0xd1]
pssub.dh s0, s2, s2
# CHECK-ASM-AND-OBJ: pssub.dw t3, a2, t3
# CHECK-ASM: encoding: [0x1b,0x6e,0xc6,0xd3]
pssub.dw t3, a2, t3
# CHECK-ASM-AND-OBJ: pssub.db a0, s0, s2
# CHECK-ASM: encoding: [0x1b,0x65,0x24,0xd5]
pssub.db a0, s0, s2
# CHECK-ASM-AND-OBJ: pasub.dh t1, a4, s0
# CHECK-ASM: encoding: [0x1b,0x63,0x87,0xd8]
pasub.dh t1, a4, s0
# CHECK-ASM-AND-OBJ: pasub.dw t1, s2, s2
# CHECK-ASM: encoding: [0x1b,0x63,0x29,0xdb]
pasub.dw t1, s2, s2
# CHECK-ASM-AND-OBJ: pasub.db a0, a0, a0
# CHECK-ASM: encoding: [0x1b,0x65,0xa5,0xdc]
pasub.db a0, a0, a0
# CHECK-ASM-AND-OBJ: pdifu.dh t5, a4, a4
# CHECK-ASM: encoding: [0x1b,0x6f,0xe7,0xe8]
pdifu.dh t5, a4, a4
# CHECK-ASM-AND-OBJ: pdifu.db t1, t1, a4
# CHECK-ASM: encoding: [0x1b,0x63,0xe3,0xec]
pdifu.db t1, t1, a4
# CHECK-ASM-AND-OBJ: pssubu.dh t5, t1, t5
# CHECK-ASM: encoding: [0x1b,0x6f,0xe3,0xf1]
pssubu.dh t5, t1, t5
# CHECK-ASM-AND-OBJ: pssubu.dw a4, a4, t1
# CHECK-ASM: encoding: [0x1b,0x67,0x67,0xf2]
pssubu.dw a4, a4, t1
# CHECK-ASM-AND-OBJ: pssubu.db s0, t5, a2
# CHECK-ASM: encoding: [0x1b,0x64,0xcf,0xf4]
pssubu.db s0, t5, a2
# CHECK-ASM-AND-OBJ: pasubu.dh t5, a2, a2
# CHECK-ASM: encoding: [0x1b,0x6f,0xc6,0xf8]
pasubu.dh t5, a2, a2
# CHECK-ASM-AND-OBJ: pasubu.dw a0, a2, a4
# CHECK-ASM: encoding: [0x1b,0x65,0xe6,0xfa]
pasubu.dw a0, a2, a4
# CHECK-ASM-AND-OBJ: pasubu.db a0, s0, s0
# CHECK-ASM: encoding: [0x1b,0x65,0x84,0xfc]
pasubu.db a0, s0, s0
# CHECK-ASM-AND-OBJ: psh1add.dh t5, a4, t5
# CHECK-ASM: encoding: [0x1b,0x6f,0xf7,0xa1]
psh1add.dh t5, a4, t5
# CHECK-ASM-AND-OBJ: psh1add.dw a4, t5, s0
# CHECK-ASM: encoding: [0x1b,0x67,0x9f,0xa2]
psh1add.dw a4, t5, s0
# CHECK-ASM-AND-OBJ: pssh1sadd.dh t3, a4, a0
# CHECK-ASM: encoding: [0x1b,0x6e,0xb7,0xb0]
pssh1sadd.dh t3, a4, a0
# CHECK-ASM-AND-OBJ: pssh1sadd.dw t1, t1, a2
# CHECK-ASM: encoding: [0x1b,0x63,0xd3,0xb2]
pssh1sadd.dw t1, t1, a2
# CHECK-ASM-AND-OBJ: ppack.dh a2, t1, s2
# CHECK-ASM: encoding: [0x1b,0xe6,0x23,0x81]
ppack.dh a2, t1, s2
# CHECK-ASM-AND-OBJ: ppack.dw t5, t3, a4
# CHECK-ASM: encoding: [0x1b,0xef,0xee,0x82]
ppack.dw t5, t3, a4
# CHECK-ASM-AND-OBJ: ppackbt.dh t1, t3, t1
# CHECK-ASM: encoding: [0x1b,0xe3,0x6e,0x90]
ppackbt.dh t1, t3, t1
# CHECK-ASM-AND-OBJ: ppackbt.dw a4, t5, a2
# CHECK-ASM: encoding: [0x1b,0xe7,0xcf,0x92]
ppackbt.dw a4, t5, a2
# CHECK-ASM-AND-OBJ: ppacktb.dh a4, t1, a2
# CHECK-ASM: encoding: [0x1b,0xe7,0xc3,0xa0]
ppacktb.dh a4, t1, a2
# CHECK-ASM-AND-OBJ: ppacktb.dw a2, t5, s0
# CHECK-ASM: encoding: [0x1b,0xe6,0x8f,0xa2]
ppacktb.dw a2, t5, s0
# CHECK-ASM-AND-OBJ: ppackt.dh a0, a0, s0
# CHECK-ASM: encoding: [0x1b,0xe5,0x85,0xb0]
ppackt.dh a0, a0, s0
# CHECK-ASM-AND-OBJ: ppackt.dw a4, a4, a2
# CHECK-ASM: encoding: [0x1b,0xe7,0xc7,0xb2]
ppackt.dw a4, a4, a2
# CHECK-ASM-AND-OBJ: pas.dhx t3, t3, s2
# CHECK-ASM: encoding: [0x1b,0xee,0x3e,0x81]
pas.dhx t3, t3, s2
# CHECK-ASM-AND-OBJ: psa.dhx a0, s2, a2
# CHECK-ASM: encoding: [0x1b,0xe5,0xd9,0x84]
psa.dhx a0, s2, a2
# CHECK-ASM-AND-OBJ: psas.dhx a2, a2, s0
# CHECK-ASM: encoding: [0x1b,0xe6,0x96,0x90]
psas.dhx a2, a2, s0
# CHECK-ASM-AND-OBJ: pssa.dhx t3, t3, t3
# CHECK-ASM: encoding: [0x1b,0xee,0xde,0x95]
pssa.dhx t3, t3, t3
# CHECK-ASM-AND-OBJ: paax.dhx t3, t3, a4
# CHECK-ASM: encoding: [0x1b,0xee,0xfe,0x98]
paax.dhx t3, t3, a4
# CHECK-ASM-AND-OBJ: pasa.dhx a0, t1, t1
# CHECK-ASM: encoding: [0x1b,0xe5,0x73,0x9c]
pasa.dhx a0, t1, t1
# CHECK-ASM-AND-OBJ: pmseq.dh a4, t1, t3
# CHECK-ASM: encoding: [0x1b,0xe7,0xd3,0xc1]
pmseq.dh a4, t1, t3
# CHECK-ASM-AND-OBJ: pmseq.dw t1, s0, a2
# CHECK-ASM: encoding: [0x1b,0xe3,0xd4,0xc2]
pmseq.dw t1, s0, a2
# CHECK-ASM-AND-OBJ: pmseq.db a2, a2, t5
# CHECK-ASM: encoding: [0x1b,0xe6,0xf6,0xc5]
pmseq.db a2, a2, t5
# CHECK-ASM-AND-OBJ: pmslt.dh s2, t5, s2
# CHECK-ASM: encoding: [0x1b,0xe9,0x3f,0xd1]
pmslt.dh s2, t5, s2
# CHECK-ASM-AND-OBJ: pmslt.dw t1, t1, a2
# CHECK-ASM: encoding: [0x1b,0xe3,0xd3,0xd2]
pmslt.dw t1, t1, a2
# CHECK-ASM-AND-OBJ: pmslt.db t5, s0, s2
# CHECK-ASM: encoding: [0x1b,0xef,0x34,0xd5]
pmslt.db t5, s0, s2
# CHECK-ASM-AND-OBJ: pmsltu.dh s2, a0, s2
# CHECK-ASM: encoding: [0x1b,0xe9,0x35,0xd9]
pmsltu.dh s2, a0, s2
# CHECK-ASM-AND-OBJ: pmsltu.dw s0, t3, a0
# CHECK-ASM: encoding: [0x1b,0xe4,0xbe,0xda]
pmsltu.dw s0, t3, a0
# CHECK-ASM-AND-OBJ: pmsltu.db s0, t3, t3
# CHECK-ASM: encoding: [0x1b,0xe4,0xde,0xdd]
pmsltu.db s0, t3, t3
# CHECK-ASM-AND-OBJ: pmin.dh a2, s0, t3
# CHECK-ASM: encoding: [0x1b,0xe6,0xd4,0xe1]
pmin.dh a2, s0, t3
# CHECK-ASM-AND-OBJ: pmin.db t3, s2, t3
# CHECK-ASM: encoding: [0x1b,0xee,0xd9,0xe5]
pmin.db t3, s2, t3
# CHECK-ASM-AND-OBJ: pminu.dh t1, t3, t5
# CHECK-ASM: encoding: [0x1b,0xe3,0xfe,0xe9]
pminu.dh t1, t3, t5
# CHECK-ASM-AND-OBJ: pminu.db t1, s0, a2
# CHECK-ASM: encoding: [0x1b,0xe3,0xd4,0xec]
pminu.db t1, s0, a2
# CHECK-ASM-AND-OBJ: pmax.dh a0, a0, a0
# CHECK-ASM: encoding: [0x1b,0xe5,0xb5,0xf0]
pmax.dh a0, a0, a0
# CHECK-ASM-AND-OBJ: pmax.db a2, a2, s2
# CHECK-ASM: encoding: [0x1b,0xe6,0x36,0xf5]
pmax.db a2, a2, s2
# CHECK-ASM-AND-OBJ: pmaxu.dh a4, t3, s0
# CHECK-ASM: encoding: [0x1b,0xe7,0x9e,0xf8]
pmaxu.dh a4, t3, s0
# CHECK-ASM-AND-OBJ: pmaxu.db a4, t5, a0
# CHECK-ASM: encoding: [0x1b,0xe7,0xbf,0xfc]
pmaxu.db a4, t5, a0
