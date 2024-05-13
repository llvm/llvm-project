# RUN: llvm-mc %s -triple=csky -show-encoding -csky-no-aliases -mattr=+e2 -mattr=+2e3 \
# RUN: -mattr=+mp1e2 | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=csky -mattr=+e2 -mattr=+2e3 -mattr=+mp1e2 < %s \
# RUN:     | llvm-objdump --mattr=+e2 --mattr=+2e3 --mattr=+mp1e2 -M no-aliases -M abi-names -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM-AND-OBJ,CHECK-OBJ %s

# CHECK-ASM-AND-OBJ: addi32 a0, sp, 2
# CHECK-ASM: encoding: [0x0e,0xe4,0x01,0x00]
addi32 a0, sp, 2

# CHECK-ASM-AND-OBJ: subi32 a0, sp, 2
# CHECK-ASM: encoding: [0x0e,0xe4,0x01,0x10]
subi32 a0, sp, 2

# CHECK-ASM-AND-OBJ: andi32 a0, sp, 2
# CHECK-ASM: encoding: [0x0e,0xe4,0x02,0x20]
andi32 a0, sp, 2

# CHECK-ASM-AND-OBJ: andni32 a0, sp, 2
# CHECK-ASM: encoding: [0x0e,0xe4,0x02,0x30]
andni32 a0, sp, 2

# CHECK-ASM-AND-OBJ: xori32 a0, sp, 2
# CHECK-ASM: encoding: [0x0e,0xe4,0x02,0x40]
xori32 a0, sp, 2

# CHECK-ASM-AND-OBJ: lsli32 a0, sp, 2
# CHECK-ASM: encoding: [0x4e,0xc4,0x20,0x48]
lsli32 a0, sp, 2

# CHECK-ASM-AND-OBJ: lsri32 a0, sp, 2
# CHECK-ASM: encoding: [0x4e,0xc4,0x40,0x48]
lsri32 a0, sp, 2

# CHECK-ASM-AND-OBJ: asri32 a0, sp, 2
# CHECK-ASM: encoding: [0x4e,0xc4,0x80,0x48]
asri32 a0, sp, 2

# CHECK-ASM-AND-OBJ: ori32 a0, sp, 2
# CHECK-ASM: encoding: [0x0e,0xec,0x02,0x00]
ori32 a0, sp, 2

# CHECK-ASM-AND-OBJ: rotli32 a0, sp, 2
# CHECK-ASM: encoding: [0x4e,0xc4,0x00,0x49]
rotli32 a0, sp, 2

# CHECK-ASM-AND-OBJ: incf32 a0, sp, 2
# CHECK-ASM: encoding: [0x0e,0xc4,0x22,0x0c]
incf32 a0, sp, 2

# CHECK-ASM-AND-OBJ: inct32 a0, sp, 2
# CHECK-ASM: encoding: [0x0e,0xc4,0x42,0x0c]
inct32 a0, sp, 2

# CHECK-ASM-AND-OBJ: decf32 a0, sp, 2
# CHECK-ASM: encoding: [0x0e,0xc4,0x82,0x0c]
decf32 a0, sp, 2

# CHECK-ASM-AND-OBJ: dect32 a0, sp, 2
# CHECK-ASM: encoding: [0x0e,0xc4,0x02,0x0d]
dect32 a0, sp, 2

# CHECK-ASM-AND-OBJ: decgt32 a0, sp, 2
# CHECK-ASM: encoding: [0x4e,0xc4,0x20,0x10]
decgt32 a0, sp, 2

# CHECK-ASM-AND-OBJ: declt32 a0, sp, 2
# CHECK-ASM: encoding: [0x4e,0xc4,0x40,0x10]
declt32 a0, sp, 2

# CHECK-ASM-AND-OBJ: decne32 a0, sp, 2
# CHECK-ASM: encoding: [0x4e,0xc4,0x80,0x10]
decne32 a0, sp, 2

# CHECK-ASM-AND-OBJ: btsti32 a0, 2
# CHECK-ASM: encoding: [0x40,0xc4,0x80,0x28]
btsti32 a0, 2

# CHECK-ASM-AND-OBJ: bclri32 a0, sp, 2
# CHECK-ASM: encoding: [0x4e,0xc4,0x20,0x28]
bclri32 a0, sp, 2

# CHECK-ASM-AND-OBJ: bseti32 a0, sp, 2
# CHECK-ASM: encoding: [0x4e,0xc4,0x40,0x28]
bseti32 a0, sp, 2

# CHECK-ASM-AND-OBJ: cmpnei32 a0, 2
# CHECK-ASM: encoding: [0x40,0xeb,0x02,0x00]
cmpnei32 a0, 2

# CHECK-ASM-AND-OBJ: cmphsi32 a0, 2
# CHECK-ASM: encoding: [0x00,0xeb,0x01,0x00]
cmphsi32 a0, 2

# CHECK-ASM-AND-OBJ: cmplti32 a0, 2
# CHECK-ASM: encoding: [0x20,0xeb,0x01,0x00]
cmplti32 a0, 2

# CHECK-ASM-AND-OBJ: movi32 a0, 2
# CHECK-ASM: encoding: [0x00,0xea,0x02,0x00]
movi32 a0, 2

# CHECK-ASM-AND-OBJ: movih32 a0, 2
# CHECK-ASM: encoding: [0x20,0xea,0x02,0x00]
movih32 a0, 2

# CHECK-ASM-AND-OBJ: addu32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x23,0x00]
addu32 a3, l0, l1

# CHECK-ASM-AND-OBJ: subu32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x83,0x00]
subu32 a3, l0, l1

# CHECK-ASM-AND-OBJ: and32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x23,0x20]
and32 a3, l0, l1

# CHECK-ASM-AND-OBJ: andn32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x43,0x20]
andn32 a3, l0, l1

# CHECK-ASM-AND-OBJ: or32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x23,0x24]
or32 a3, l0, l1

# CHECK-ASM-AND-OBJ: xor32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x43,0x24]
xor32 a3, l0, l1

# CHECK-ASM-AND-OBJ: nor32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x83,0x24]
nor32 a3, l0, l1

# CHECK-ASM-AND-OBJ: lsl32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x23,0x40]
lsl32 a3, l0, l1

# CHECK-ASM-AND-OBJ: rotl32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x03,0x41]
rotl32 a3, l0, l1

# CHECK-ASM-AND-OBJ: lsr32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x43,0x40]
lsr32 a3, l0, l1

# CHECK-ASM-AND-OBJ: asr32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x83,0x40]
asr32 a3, l0, l1

# CHECK-ASM-AND-OBJ: lslc32 a0, sp, 2
# CHECK-ASM: encoding: [0x2e,0xc4,0x20,0x4c]
lslc32 a0, sp, 2

# CHECK-ASM-AND-OBJ: lsrc32 a0, sp, 2
# CHECK-ASM: encoding: [0x2e,0xc4,0x40,0x4c]
lsrc32 a0, sp, 2

# CHECK-ASM-AND-OBJ: asrc32 a0, sp, 2
# CHECK-ASM: encoding: [0x2e,0xc4,0x80,0x4c]
asrc32 a0, sp, 2

# CHECK-ASM-AND-OBJ: xsr32 a0, sp, 2
# CHECK-ASM: encoding: [0x2e,0xc4,0x00,0x4d]
xsr32 a0, sp, 2

# CHECK-ASM-AND-OBJ: bmaski32 a3, 17
# CHECK-ASM: encoding: [0x00,0xc6,0x23,0x50]
bmaski32 a3, 17

# CHECK-ASM-AND-OBJ: mult32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x23,0x84]
mult32 a3, l0, l1

# CHECK-ASM-AND-OBJ: divs32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x43,0x80]
divs32 a3, l0, l1

# CHECK-ASM-AND-OBJ: divu32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x23,0x80]
divu32 a3, l0, l1

# CHECK-ASM-AND-OBJ: ixh32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x23,0x08]
ixh32 a3, l0, l1

# CHECK-ASM-AND-OBJ: ixw32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x43,0x08]
ixw32 a3, l0, l1

# CHECK-ASM-AND-OBJ: ixd32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x83,0x08]
ixd32 a3, l0, l1

# CHECK-ASM-AND-OBJ: addc32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x43,0x00]
addc32 a3, l0, l1

# CHECK-ASM-AND-OBJ: subc32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x03,0x01]
subc32 a3, l0, l1

# CHECK-OBJ: ld32.b a0, (sp, 0x2)
# CHECK-ASM: ld32.b a0, (sp, 2)
# CHECK-ASM: encoding: [0x0e,0xd8,0x02,0x00]
ld32.b a0, (sp, 2)

# CHECK-OBJ: ld32.bs a0, (sp, 0x2)
# CHECK-ASM: ld32.bs a0, (sp, 2)
# CHECK-ASM: encoding: [0x0e,0xd8,0x02,0x40]
ld32.bs a0, (sp, 2)

# CHECK-OBJ: ld32.h a0, (sp, 0x2)
# CHECK-ASM: ld32.h a0, (sp, 2)
# CHECK-ASM: encoding: [0x0e,0xd8,0x01,0x10]
ld32.h a0, (sp, 2)

# CHECK-OBJ: ld32.hs a0, (sp, 0x2)
# CHECK-ASM: ld32.hs a0, (sp, 2)
# CHECK-ASM: encoding: [0x0e,0xd8,0x01,0x50]
ld32.hs a0, (sp, 2)

# CHECK-OBJ: ld32.w a0, (sp, 0x4)
# CHECK-ASM: ld32.w a0, (sp, 4)
# CHECK-ASM: encoding: [0x0e,0xd8,0x01,0x20]
ld32.w a0, (sp, 4)

# CHECK-ASM-AND-OBJ: ldr32.b a0, (sp, l1 << 2)
# CHECK-ASM: encoding: [0xae,0xd0,0x80,0x00]
ldr32.b a0, (sp, l1 << 2)

# CHECK-OBJ: ldex32.w a0, (sp, 0x4)
# CHECK-ASM: ldex32.w a0, (sp, 4)
# CHECK-ASM: encoding: [0x0e,0xd8,0x01,0x70]
ldex32.w a0, (sp, 4)

# CHECK-ASM-AND-OBJ: ldr32.bs a0, (sp, l1 << 2)
# CHECK-ASM: encoding: [0xae,0xd0,0x80,0x10]
ldr32.bs a0, (sp, l1 << 2)

# CHECK-ASM-AND-OBJ: ldr32.h a0, (sp, l1 << 3)
# CHECK-ASM: encoding: [0xae,0xd0,0x00,0x05]
ldr32.h a0, (sp, l1 << 3)

# CHECK-ASM-AND-OBJ: ldr32.hs a0, (sp, l1 << 3)
# CHECK-ASM: encoding: [0xae,0xd0,0x00,0x15]
ldr32.hs a0, (sp, l1 << 3)

# CHECK-ASM-AND-OBJ: ldr32.w a0, (sp, l1 << 3)
# CHECK-ASM: encoding: [0xae,0xd0,0x00,0x09]
ldr32.w a0, (sp, l1 << 3)

# CHECK-OBJ: st32.b a0, (sp, 0x2)
# CHECK-ASM: st32.b a0, (sp, 2)
# CHECK-ASM: encoding: [0x0e,0xdc,0x02,0x00]
st32.b a0, (sp, 2)

# CHECK-OBJ: st32.h a0, (sp, 0x2)
# CHECK-ASM: st32.h a0, (sp, 2)
# CHECK-ASM: encoding: [0x0e,0xdc,0x01,0x10]
st32.h a0, (sp, 2)

# CHECK-OBJ: st32.w a0, (sp, 0x4)
# CHECK-ASM: st32.w a0, (sp, 4)
# CHECK-ASM: encoding: [0x0e,0xdc,0x01,0x20]
st32.w a0, (sp, 4)

# CHECK-OBJ: stex32.w a0, (sp, 0x4)
# CHECK-ASM: stex32.w a0, (sp, 4)
# CHECK-ASM: encoding: [0x0e,0xdc,0x01,0x70]
stex32.w a0, (sp, 4)

# CHECK-ASM-AND-OBJ: str32.b a0, (sp, l1 << 2)
# CHECK-ASM: encoding: [0xae,0xd4,0x80,0x00]
str32.b a0, (sp, l1 << 2)

# CHECK-ASM-AND-OBJ: str32.h a0, (sp, l1 << 3)
# CHECK-ASM: encoding: [0xae,0xd4,0x00,0x05]
str32.h a0, (sp, l1 << 3)

# CHECK-ASM-AND-OBJ: str32.w a0, (sp, l1 << 3)
# CHECK-ASM: encoding: [0xae,0xd4,0x00,0x09]
str32.w a0, (sp, l1 << 3)

# CHECK-ASM-AND-OBJ: ldm32  a1-a2, (a0)
# CHECK-ASM: encoding: [0x20,0xd0,0x21,0x1c]
ldm32  a1-a2, (a0)

# CHECK-ASM-AND-OBJ: stm32  a1-a2, (a0)
# CHECK-ASM: encoding: [0x20,0xd4,0x21,0x1c]
stm32  a1-a2, (a0)

# CHECK-ASM-AND-OBJ: ldm32  l0-l3, (a0)
# CHECK-ASM: encoding: [0x80,0xd0,0x23,0x1c]
ldq32  r4-r7, (a0)

# CHECK-ASM-AND-OBJ: stm32  l0-l3, (a0)
# CHECK-ASM: encoding: [0x80,0xd4,0x23,0x1c]
stq32  r4-r7, (a0)

# CHECK-ASM-AND-OBJ: brev32 a3, l0
# CHECK-ASM: encoding: [0x04,0xc4,0x03,0x62]
brev32 a3, l0

# CHECK-ASM-AND-OBJ: abs32 a3, l0
# CHECK-ASM: encoding: [0x04,0xc4,0x03,0x02]
abs32 a3, l0

# CHECK-ASM-AND-OBJ: bgenr32 a3, l0
# CHECK-ASM: encoding: [0x04,0xc4,0x43,0x50]
bgenr32 a3, l0

# CHECK-ASM-AND-OBJ: revb32 a3, l0
# CHECK-ASM: encoding: [0x04,0xc4,0x83,0x60]
revb32 a3, l0

# CHECK-ASM-AND-OBJ: revh32 a3, l0
# CHECK-ASM: encoding: [0x04,0xc4,0x03,0x61]
revh32 a3, l0

# CHECK-ASM-AND-OBJ: ff0.32 a3, l0
# CHECK-ASM: encoding: [0x04,0xc4,0x23,0x7c]
ff0.32 a3, l0

# CHECK-ASM-AND-OBJ: ff1.32 a3, l0
# CHECK-ASM: encoding: [0x04,0xc4,0x43,0x7c]
ff1.32 a3, l0

# CHECK-ASM-AND-OBJ: xtrb0.32 a3, l0
# CHECK-ASM: encoding: [0x04,0xc4,0x23,0x70]
xtrb0.32 a3, l0

# CHECK-ASM-AND-OBJ: xtrb1.32 a3, l0
# CHECK-ASM: encoding: [0x04,0xc4,0x43,0x70]
xtrb1.32 a3, l0

# CHECK-ASM-AND-OBJ: xtrb2.32 a3, l0
# CHECK-ASM: encoding: [0x04,0xc4,0x83,0x70]
xtrb2.32 a3, l0

# CHECK-ASM-AND-OBJ: xtrb3.32 a3, l0
# CHECK-ASM: encoding: [0x04,0xc4,0x03,0x71]
xtrb3.32 a3, l0

# CHECK-ASM-AND-OBJ: mvc32 a3
# CHECK-ASM: encoding: [0x00,0xc4,0x03,0x05]
mvc32 a3

# CHECK-ASM-AND-OBJ: mvcv32 a3
# CHECK-ASM: encoding: [0x00,0xc4,0x03,0x06]
mvcv32 a3

# CHECK-ASM-AND-OBJ: cmpne32 a3, l0
# CHECK-ASM: encoding: [0x83,0xc4,0x80,0x04]
cmpne32 a3, l0

# CHECK-ASM-AND-OBJ: cmphs32 a3, l0
# CHECK-ASM: encoding: [0x83,0xc4,0x20,0x04]
cmphs32 a3, l0

# CHECK-ASM-AND-OBJ: cmplt32 a3, l0
# CHECK-ASM: encoding: [0x83,0xc4,0x40,0x04]
cmplt32 a3, l0

# CHECK-ASM-AND-OBJ: zext32 a3, l0, 7, 0
# CHECK-ASM: encoding: [0x04,0xc4,0xe3,0x54]
zext32 a3, l0, 7, 0

# CHECK-ASM-AND-OBJ: sext32 a3, l0, 7, 0
# CHECK-ASM: encoding: [0x04,0xc4,0xe3,0x58]
sext32 a3, l0, 7, 0

# CHECK-ASM-AND-OBJ: ldm32 l1-l3, (a0)
# CHECK-ASM: encoding: [0xa0,0xd0,0x22,0x1c]
ldm32 r5-r7, (a0)

# CHECK-ASM-AND-OBJ: stm32 l1-l3, (a0)
# CHECK-ASM: encoding: [0xa0,0xd4,0x22,0x1c]
stm32 r5-r7, (a0)

# CHECK-ASM-AND-OBJ: setc32
# CHECK-ASM: encoding: [0x00,0xc4,0x20,0x04]
setc32

# CHECK-ASM-AND-OBJ: clrc32
# CHECK-ASM: encoding: [0x00,0xc4,0x80,0x04]
clrc32

# CHECK-ASM-AND-OBJ: tst32 a3, l0
# CHECK-ASM: encoding: [0x83,0xc4,0x80,0x20]
tst32 a3, l0

# CHECK-ASM-AND-OBJ: tstnbz32 a3
# CHECK-ASM: encoding: [0x03,0xc4,0x00,0x21]
tstnbz32 a3

# CHECK-ASM-AND-OBJ: clrf32 a3
# CHECK-ASM: encoding: [0x60,0xc4,0x20,0x2c]
clrf32 a3

# CHECK-ASM-AND-OBJ: clrt32 a3
# CHECK-ASM: encoding: [0x60,0xc4,0x40,0x2c]
clrt32 a3

# CHECK-ASM-AND-OBJ: bar.brwarw
# CHECK-ASM: encoding: [0x00,0xc0,0x2f,0x84]
bar.brwarw

# CHECK-ASM-AND-OBJ: bar.brwarws
# CHECK-ASM: encoding: [0x00,0xc2,0x2f,0x84]
bar.brwarws

# CHECK-ASM-AND-OBJ: bar.brarw
# CHECK-ASM: encoding: [0x00,0xc0,0x27,0x84]
bar.brarw

# CHECK-ASM-AND-OBJ: bar.brarws
# CHECK-ASM: encoding: [0x00,0xc2,0x27,0x84]
bar.brarws

# CHECK-ASM-AND-OBJ: bar.brwaw
# CHECK-ASM: encoding: [0x00,0xc0,0x2e,0x84]
bar.brwaw

# CHECK-ASM-AND-OBJ: bar.brwaws
# CHECK-ASM: encoding: [0x00,0xc2,0x2e,0x84]
bar.brwaws

# CHECK-ASM-AND-OBJ: bar.brar
# CHECK-ASM: encoding: [0x00,0xc0,0x25,0x84]
bar.brar

# CHECK-ASM-AND-OBJ: bar.brars
# CHECK-ASM: encoding: [0x00,0xc2,0x25,0x84]
bar.brars

# CHECK-ASM-AND-OBJ: bar.bwaw
# CHECK-ASM: encoding: [0x00,0xc0,0x2a,0x84]
bar.bwaw

# CHECK-ASM-AND-OBJ: bar.bwaws
# CHECK-ASM: encoding: [0x00,0xc2,0x2a,0x84]
bar.bwaws

# CHECK-ASM-AND-OBJ: sync32
# CHECK-ASM: encoding: [0x00,0xc0,0x20,0x04]
sync32

# CHECK-ASM-AND-OBJ: sync32.s
# CHECK-ASM: encoding: [0x00,0xc2,0x20,0x04]
sync32.s

# CHECK-ASM-AND-OBJ: sync32.i
# CHECK-ASM: encoding: [0x20,0xc0,0x20,0x04]
sync32.i

# CHECK-ASM-AND-OBJ: sync32.is
# CHECK-ASM: encoding: [0x20,0xc2,0x20,0x04]
sync32.is

# CHECK-ASM-AND-OBJ: rfi32
# CHECK-ASM: encoding: [0x00,0xc0,0x20,0x44]
rfi32

# CHECK-ASM-AND-OBJ: stop32
# CHECK-ASM: encoding: [0x00,0xc0,0x20,0x48]
stop32

# CHECK-ASM-AND-OBJ: wait32
# CHECK-ASM: encoding: [0x00,0xc0,0x20,0x4c]
wait32

# CHECK-ASM-AND-OBJ: doze32
# CHECK-ASM: encoding: [0x00,0xc0,0x20,0x50]
doze32

# CHECK-ASM: br32 .L.test
# CHECK-ASM: encoding: [A,0xe8'A',A,A]
# CHECK-ASM: fixup A - offset: 0, value: .L.test, kind: fixup_csky_pcrel_imm16_scale2
.L.test:
br32 .L.test

# CHECK-ASM: bt32 .L.test2
# CHECK-ASM: encoding: [0x60'A',0xe8'A',A,A]
# CHECK-ASM: fixup A - offset: 0, value: .L.test2, kind: fixup_csky_pcrel_imm16_scale2
.L.test2:
bt32 .L.test2

# CHECK-ASM: bf32 .L.test3
# CHECK-ASM: encoding: [0x40'A',0xe8'A',A,A]
# CHECK-ASM: fixup A - offset: 0, value: .L.test3, kind: fixup_csky_pcrel_imm16_scale2
.L.test3:
bf32 .L.test3

# CHECK-ASM: bez32 a0, .L.test4
# CHECK-ASM: encoding: [A,0xe9'A',A,A]
# CHECK-ASM: fixup A - offset: 0, value: .L.test4, kind: fixup_csky_pcrel_imm16_scale2
.L.test4:
bez32 a0, .L.test4

# CHECK-ASM: bnez32 a0, .L.test5
# CHECK-ASM: encoding: [0x20'A',0xe9'A',A,A]
# CHECK-ASM: fixup A - offset: 0, value: .L.test5, kind: fixup_csky_pcrel_imm16_scale2
.L.test5:
bnez32 a0, .L.test5

# CHECK-ASM: bhz32 a0, .L.test6
# CHECK-ASM: encoding: [0x40'A',0xe9'A',A,A]
# CHECK-ASM: fixup A - offset: 0, value: .L.test6, kind: fixup_csky_pcrel_imm16_scale2
.L.test6:
bhz32 a0, .L.test6

# CHECK-ASM: blsz32 a0, .L.test7
# CHECK-ASM: encoding: [0x60'A',0xe9'A',A,A]
# CHECK-ASM: fixup A - offset: 0, value: .L.test7, kind: fixup_csky_pcrel_imm16_scale2
.L.test7:
blsz32 a0, .L.test7

# CHECK-ASM: blz32 a0, .L.test8
# CHECK-ASM: encoding: [0x80'A',0xe9'A',A,A]
# CHECK-ASM: fixup A - offset: 0, value: .L.test8, kind: fixup_csky_pcrel_imm16_scale2
.L.test8:
blz32 a0, .L.test8

# CHECK-ASM: bhsz32 a0, .L.test9
# CHECK-ASM: encoding: [0xa0'A',0xe9'A',A,A]
# CHECK-ASM: fixup A - offset: 0, value: .L.test9, kind: fixup_csky_pcrel_imm16_scale2
.L.test9:
bhsz32 a0, .L.test9

# CHECK-ASM: jmp32 a3
# CHECK-ASM: encoding: [0xc3,0xe8,0x00,0x00]
jmp32 a3

# CHECK-ASM: jmpi32 [.L.test10]
# CHECK-ASM: encoding: [0xc0'A',0xea'A',A,A]
# CHECK-ASM: fixup A - offset: 0, value: .L.test10, kind: fixup_csky_pcrel_uimm16_scale4
.L.test10:
jmpi32 [.L.test10]

# CHECK-ASM: bsr32 .L.test11
# CHECK-ASM: encoding: [A,0xe0'A',A,A]
# CHECK-ASM: fixup A - offset: 0, value: .L.test11, kind: fixup_csky_pcrel_imm26_scale2
.L.test11:
bsr32 .L.test11

# CHECK-ASM: jsr32 a3
# CHECK-ASM: encoding: [0xe3,0xe8,0x00,0x00]
jsr32 a3

# CHECK-ASM: jsri32 [.L.test12]
# CHECK-ASM: encoding: [0xe0'A',0xea'A',A,A]
# CHECK-ASM: fixup A - offset: 0, value: .L.test12, kind: fixup_csky_pcrel_uimm16_scale4
.L.test12:
jsri32 [.L.test12]

# CHECK-ASM: grs32 a0, .L.test13
# CHECK-ASM: encoding: [0x0c'A',0xcc'A',A,A]
# CHECK-ASM: fixup A - offset: 0, value: .L.test13, kind: fixup_csky_pcrel_imm18_scale2
.L.test13:
grs32 a0, .L.test13

# CHECK-ASM: lrw32 a0, [.L.test14]
# CHECK-ASM: encoding: [0x80'A',0xea'A',A,A]
# CHECK-ASM: fixup A - offset: 0, value: .L.test14, kind: fixup_csky_pcrel_uimm16_scale4
.L.test14:
lrw32 a0, [.L.test14]

# RUN: not llvm-mc -triple csky -mattr=+e2 -mattr=+2e3 --defsym=ERR=1 < %s 2>&1 | FileCheck %s

.ifdef ERR

# uimm12/oimm12/nimm12
addi32 a0, a0, 0 # CHECK: :[[#@LINE]]:16: error: immediate must be an integer in the range [1, 4096]
subi32 a0, a0, 4097 # CHECK: :[[#@LINE]]:16: error: immediate must be an integer in the range [1, 4096]
andi32 a0, a0, 4096 # CHECK: :[[#@LINE]]:16: error: immediate must be an integer in the range [0, 4095]
andni32 a0, a0, 4096 # CHECK: :[[#@LINE]]:17: error: immediate must be an integer in the range [0, 4095]
xori32 a0, a0, 4096 # CHECK: :[[#@LINE]]:16: error: immediate must be an integer in the range [0, 4095]


# uimm5
lsli32 a0, a0, 32 # CHECK: :[[#@LINE]]:16: error: immediate must be an integer in the range [0, 31]
lsri32 a0, a0, 32 # CHECK: :[[#@LINE]]:16: error: immediate must be an integer in the range [0, 31]
asri32 a0, a0, 32 # CHECK: :[[#@LINE]]:16: error: immediate must be an integer in the range [0, 31]
rotli32 a0, a0, 32 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 31]

# uimm12(shift0)/uimm12_1/uimm12_2
ld32.b a0, (a0, -1)  # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 4095]
ld32.h a0, (a0, 4095)  # CHECK: :[[@LINE]]:17: error: immediate must be a multiple of 2 bytes in the range [0, 4094]
ld32.h a0, (a0, 4093)  # CHECK: :[[@LINE]]:17: error: immediate must be a multiple of 2 bytes in the range [0, 4094]
ld32.w a0, (a0, 4093)  # CHECK: :[[@LINE]]:17: error: immediate must be a multiple of 4 bytes in the range [0, 4092]
ld32.w a0, (a0, 2)  # CHECK: :[[@LINE]]:17: error: immediate must be a multiple of 4 bytes in the range [0, 4092]
ldr32.b a0, (a0, a1 << 4) # CHECK: :[[@LINE]]:24: error: immediate must be an integer in the range [0, 3]
ldr32.bs a0, (a0, a1 << 4) # CHECK: :[[@LINE]]:25: error: immediate must be an integer in the range [0, 3]
ldr32.h a0, (a0, a1 << -1) # CHECK: :[[@LINE]]:24: error: immediate must be an integer in the range [0, 3]
ldr32.hs a0, (a0, a1 << 4) # CHECK: :[[@LINE]]:25: error: immediate must be an integer in the range [0, 3]
ldr32.w a0, (a0, a1 << 4) # CHECK: :[[@LINE]]:24: error: immediate must be an integer in the range [0, 3]

st32.b a0, (a0, -1)  # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 4095]
st32.h a0, (a0, 4095)  # CHECK: :[[@LINE]]:17: error: immediate must be a multiple of 2 bytes in the range [0, 4094]
st32.h a0, (a0, 4093)  # CHECK: :[[@LINE]]:17: error: immediate must be a multiple of 2 bytes in the range [0, 4094]
st32.w a0, (a0, 4093)  # CHECK: :[[@LINE]]:17: error: immediate must be a multiple of 4 bytes in the range [0, 4092]
st32.w a0, (a0, 2)  # CHECK: :[[@LINE]]:17: error: immediate must be a multiple of 4 bytes in the range [0, 4092]

str32.b a0, (a0, a1 << 4) # CHECK: :[[@LINE]]:24: error: immediate must be an integer in the range [0, 3]
str32.h a0, (a0, a1 << -1) # CHECK: :[[@LINE]]:24: error: immediate must be an integer in the range [0, 3]
str32.w a0, (a0, a1 << 4) # CHECK: :[[@LINE]]:24: error: immediate must be an integer in the range [0, 3]

# uimm16/oimm16
ori32 a0, a0, 0x10000 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 65535]
cmpnei32 a0, 0x10000 # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 65535]
cmphsi32 a0, 0x10001 # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [1, 65536]
cmplti32 a0, 0 # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [1, 65536]
movi32 a0, 0x10000 # CHECK: :[[@LINE]]:12: error: immediate must be an integer in the range [0, 65535]
movih32 a0, 0x10000 # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 65535]

# Invalid mnemonics
subs t0, t2, t1 # CHECK: :[[#@LINE]]:1: error: unrecognized instruction mnemonic
nandi t0, t2, 0 # CHECK: :[[#@LINE]]:1: error: unrecognized instruction mnemonic

# Invalid register names
addi32 foo, sp, 10 # CHECK: :[[#@LINE]]:8: error: unknown operand
lsli32 a10, a2, 0x20 # CHECK: :[[#@LINE]]:8: error: unknown operand
asri32 x32, s0, s0 # CHECK: :[[#@LINE]]:8: error: unknown operand

# Invalid operand types
xori32 sp, 22, 220 # CHECK: :[[#@LINE]]:12: error: invalid operand for instruction
subu32 t0, t2, 1 # CHECK: :[[#@LINE]]:16: error: invalid operand for instruction

# Too many operands
subi32 t2, t3, 0x50, 0x60 # CHECK: :[[@LINE]]:22: error: invalid operand for instruction

# Too few operands
xori32 a0, a1 # CHECK: :[[#@LINE]]:1: error: too few operands for instruction
xor32 a0, a2 # CHECK: :[[#@LINE]]:1: error: too few operands for instruction

ldm32 a1-a33, (a1) # CHECK: :[[#@LINE]]:10: error: invalid register
stm32 a1-a33, (a1) # CHECK: :[[#@LINE]]:10: error: invalid register

ldq32 a1-a2, (a1) # CHECK: :[[#@LINE]]:1: error: Register sequence is not valid. 'r4-r7' expected
stq32 a1-a3, (a1) # CHECK: :[[#@LINE]]:1: error: Register sequence is not valid. 'r4-r7' expected

.endif