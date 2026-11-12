# RUN: llvm-mc -filetype=asm -triple=hexagon-unknown-elf -mcpu=hexagonv67t %s | FileCheck %s
# RUN: llvm-mc -filetype=asm -triple=hexagon-unknown-elf -mcpu=hexagonv67 -mattr=+audio %s | FileCheck %s
# RUN: not llvm-mc -filetype=asm -triple=hexagon-unknown-elf -mcpu=hexagonv67 %s 2>&1 | FileCheck -check-prefix=CHECKINV %s

# CHECK: clip
# CHECKINV: error: invalid instruction
r0 = clip(r0, #1)

# CHECK: cround
# CHECKINV: error: invalid instruction
r1:0 = cround(r1:0, #4)

# CHECK: vclip
# CHECKINV: error: invalid instruction
r1:0 = vclip(r1:0, #2)

# CHECK: += cmpyiw
# CHECKINV: error: invalid instruction
r5:4 += cmpyiw(r5:4,r3:2)

# CHECK: cmpyrw
# CHECKINV: error: invalid instruction
r5:4 = cmpyrw(r5:4,r3:2)

# CHECK: cmpyrw(r7:6,r5:4):<<1:rnd:sat
# CHECKINV: error: invalid instruction
r7 = cmpyrw(r7:6,r5:4):<<1:rnd:sat
