# RUN: llvm-mc %s -triple=csky -show-encoding -csky-no-aliases -enable-csky-asm-compressed-inst=true -mattr=+e1 \
# RUN: -mattr=+e2 -mattr=+2e3 -mattr=+btst16 | FileCheck -check-prefixes=CHECK-ASM %s

# CHECK-ASM: addu16 a0, a1
# CHECK-ASM: encoding: [0x04,0x60]
addu32 a0, a0, a1

# CHECK-ASM: addu16 a0, a1
# CHECK-ASM: encoding: [0x04,0x60]
addu32 a0, a1, a0

# CHECK-ASM: addu16 a0, a1, a2
# CHECK-ASM: encoding: [0x08,0x59]
addu32 a0, a1, a2

# CHECK-ASM: subu16 a0, a1
# CHECK-ASM: encoding: [0x06,0x60]
subu32 a0, a0, a1

# CHECK-ASM: subu16 a0, a1, a2
# CHECK-ASM: encoding: [0x09,0x59]
subu32 a0, a1, a2

# CHECK-ASM: addc16 a0, a1
# CHECK-ASM: encoding: [0x05,0x60]
addc32 a0, a0, a1

# CHECK-ASM: subc16 a0, a1
# CHECK-ASM: encoding: [0x07,0x60]
subc32 a0, a0, a1

# CHECK-ASM: addi16 a0, a1, 1
# CHECK-ASM: encoding: [0x02,0x59]
addi32 a0, a1, 1

# CHECK-ASM: subi16 a0, a1, 1
# CHECK-ASM: encoding: [0x03,0x59]
subi32 a0, a1, 1

# CHECK-ASM: addi16 a0, 10
# CHECK-ASM: encoding: [0x09,0x20]
addi32 a0, a0, 10

# CHECK-ASM: subi16 a0, 10
# CHECK-ASM: encoding: [0x09,0x28]
subi32 a0, a0, 10

# CHECK-ASM: addi16 sp, sp, 4
# CHECK-ASM: encoding: [0x01,0x14]
addi32 sp, sp, 4

# CHECK-ASM: subi16 sp, sp, 4
# CHECK-ASM: encoding: [0x21,0x14]
subi32 sp, sp, 4

# CHECK-ASM: addi16 a0, sp, 4
# CHECK-ASM: encoding: [0x01,0x18]
addi32 a0, sp, 4

# CHECK-ASM: mult16 a0, a1
# CHECK-ASM: encoding: [0x04,0x7c]
mult32 a0, a0, a1

# CHECK-ASM: mult16 a0, a1
# CHECK-ASM: encoding: [0x04,0x7c]
mult32 a0, a1, a0

# CHECK-ASM: and16 a0, a1
# CHECK-ASM: encoding: [0x04,0x68]
and32 a0, a0, a1

# CHECK-ASM: and16 a0, a1
# CHECK-ASM: encoding: [0x04,0x68]
and32 a0, a1, a0

# CHECK-ASM: or16 a0, a1
# CHECK-ASM: encoding: [0x04,0x6c]
or32 a0, a0, a1

# CHECK-ASM: or16 a0, a1
# CHECK-ASM: encoding: [0x04,0x6c]
or32 a0, a1, a0

# CHECK-ASM: xor16 a0, a1
# CHECK-ASM: encoding: [0x05,0x6c]
xor32 a0, a0, a1

# CHECK-ASM: xor16 a0, a1
# CHECK-ASM: encoding: [0x05,0x6c]
xor32 a0, a1, a0

# CHECK-ASM: andn16 a0, a1
# CHECK-ASM: encoding: [0x05,0x68]
andn32 a0, a0, a1

# CHECK-ASM: nor16 a0, a1
# CHECK-ASM: encoding: [0x06,0x6c]
nor32 a0, a0, a1

# CHECK-ASM: lsl16 a0, a1
# CHECK-ASM: encoding: [0x04,0x70]
lsl32 a0, a0, a1

# CHECK-ASM: lsr16 a0, a1
# CHECK-ASM: encoding: [0x05,0x70]
lsr32 a0, a0, a1

# CHECK-ASM: asr16 a0, a1
# CHECK-ASM: encoding: [0x06,0x70]
asr32 a0, a0, a1

# CHECK-ASM: rotl16 a0, a1
# CHECK-ASM: encoding: [0x07,0x70]
rotl32 a0, a0, a1

# CHECK-ASM: revb16 a0, a1
# CHECK-ASM: encoding: [0x06,0x78]
revb32 a0, a1

# CHECK-ASM: lsli16 a0, a1, 2
# CHECK-ASM: encoding: [0x02,0x41]
lsli32 a0, a1, 2

# CHECK-ASM: lsri16 a0, a1, 2
# CHECK-ASM: encoding: [0x02,0x49]
lsri32 a0, a1, 2

# CHECK-ASM: asri16 a0, a1, 2
# CHECK-ASM: encoding: [0x02,0x51]
asri32 a0, a1, 2

# CHECK-ASM: cmphs16 a0, a1
# CHECK-ASM: encoding: [0x40,0x64]
cmphs32 a0, a1

# CHECK-ASM: cmplt16 a0, a1
# CHECK-ASM: encoding: [0x41,0x64]
cmplt32 a0, a1

# CHECK-ASM: cmpne16 a0, a1
# CHECK-ASM: encoding: [0x42,0x64]
cmpne32 a0, a1

# CHECK-ASM: cmphsi16 a0, 1
# CHECK-ASM: encoding: [0x00,0x38]
cmphsi32 a0, 1

# CHECK-ASM: cmplti16 a0, 1
# CHECK-ASM: encoding: [0x20,0x38]
cmplti32 a0, 1

# CHECK-ASM: cmpnei16 a0, 1
# CHECK-ASM: encoding: [0x41,0x38]
cmpnei32 a0, 1

# CHECK-ASM: jsr16 a0
# CHECK-ASM: encoding: [0xc1,0x7b]
jsr32 a0

# CHECK-ASM: mvcv16 a0
# CHECK-ASM: encoding: [0x03,0x64]
mvcv32 a0

# CHECK-ASM: movi16 a0, 1
# CHECK-ASM: encoding: [0x01,0x30]
movi32 a0, 1

# CHECK-ASM: ld16.b a0, (a1, 1)
# CHECK-ASM: encoding: [0x01,0x81]
ld32.b a0, (a1, 1)

# CHECK-ASM: ld16.h a0, (a1, 2)
# CHECK-ASM: encoding: [0x01,0x89]
ld32.h a0, (a1, 2)

# CHECK-ASM: ld16.w a0, (a1, 4)
# CHECK-ASM: encoding: [0x01,0x91]
ld32.w a0, (a1, 4)

# CHECK-ASM: ld16.w a0, (sp, 4)
# CHECK-ASM: encoding: [0x01,0x98]
ld32.w a0, (sp, 4)

# CHECK-ASM: st16.b a0, (a1, 1)
# CHECK-ASM: encoding: [0x01,0xa1]
st32.b a0, (a1, 1)

# CHECK-ASM: st16.h a0, (a1, 2)
# CHECK-ASM: encoding: [0x01,0xa9]
st32.h a0, (a1, 2)

# CHECK-ASM: st16.w a0, (a1, 4)
# CHECK-ASM: encoding: [0x01,0xb1]
st32.w a0, (a1, 4)

# CHECK-ASM: st16.w a0, (sp, 4)
# CHECK-ASM: encoding: [0x01,0xb8]
st32.w a0, (sp, 4)

# CHECK-ASM: btsti16 a0, 1
# CHECK-ASM: encoding: [0xc1,0x38]
btsti32 a0, 1

# CHECK-ASM: bclri16 a0, 1
# CHECK-ASM: encoding: [0x81,0x38]
bclri32 a0, a0, 1

# CHECK-ASM: bseti16 a0, 1
# CHECK-ASM: encoding: [0xa1,0x38]
bseti32 a0, a0, 1
