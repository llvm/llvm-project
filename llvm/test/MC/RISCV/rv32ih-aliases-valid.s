# RUN: llvm-mc %s -triple=riscv32 -mattr=+h -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST,CHECK-ALIAS-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+h -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST,CHECK-ALIAS-INST %s
# RUN: llvm-mc -filetype=obj -mattr=+h -triple riscv32 < %s \
# RUN:     | llvm-objdump --mattr=+h -M no-aliases -d - \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-NOALIAS-INST %s
# RUN: llvm-mc -filetype=obj -mattr=+h -triple riscv64 < %s \
# RUN:     | llvm-objdump --mattr=+h -M no-aliases -d - \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-NOALIAS-INST %s

# CHECK-ALIAS-INST: hfence.gvma
# CHECK-NOALIAS-INST: hfence.gvma zero, zero
# CHECK: encoding: [0x73,0x00,0x00,0x62]
hfence.gvma

# CHECK-ALIAS-INST: hfence.gvma a0
# CHECK-NOALIAS-INST: hfence.gvma a0, zero
# CHECK: encoding: [0x73,0x00,0x05,0x62]
hfence.gvma a0

# CHECK-ALIAS-INST: hfence.vvma
# CHECK-NOALIAS-INST: hfence.vvma zero, zero
# CHECK: encoding: [0x73,0x00,0x00,0x22]
hfence.vvma

# CHECK-ALIAS-INST: hfence.vvma a0
# CHECK-NOALIAS-INST: hfence.vvma a0, zero
# CHECK: encoding: [0x73,0x00,0x05,0x22]
hfence.vvma a0

# CHECK-INST: hlv.b a0, (a1)
# CHECK: encoding: [0x73,0xc5,0x05,0x60]
hlv.b   a0, 0(a1)

# CHECK-INST: hlv.bu a0, (a1)
# CHECK: encoding: [0x73,0xc5,0x15,0x60]
hlv.bu  a0, 0(a1)

# CHECK-INST: hlv.h a1, (a2)
# CHECK: encoding: [0xf3,0x45,0x06,0x64]
hlv.h   a1, 0(a2)

# CHECK-INST: hlv.hu a1, (a1)
# CHECK: encoding: [0xf3,0xc5,0x15,0x64]
hlv.hu  a1, 0(a1)

# CHECK-INST: hlvx.hu a1, (a2)
# CHECK: encoding: [0xf3,0x45,0x36,0x64]
hlvx.hu a1, 0(a2)

# CHECK-INST: hlv.w a2, (a2)
# CHECK: encoding: [0x73,0x46,0x06,0x68]
hlv.w   a2, 0(a2)

# CHECK-INST: hlvx.wu a2, (a3)
# CHECK: encoding: [0x73,0xc6,0x36,0x68]
hlvx.wu a2, 0(a3)

# CHECK-INST: hsv.b a0, (a1)
# CHECK: encoding: [0x73,0xc0,0xa5,0x62]
hsv.b   a0, 0(a1)

# CHECK-INST: hsv.h a0, (a1)
# CHECK: encoding: [0x73,0xc0,0xa5,0x66]
hsv.h   a0, 0(a1)

# CHECK-INST: hsv.w a0, (a1)
# CHECK: encoding: [0x73,0xc0,0xa5,0x6a]
hsv.w   a0, 0(a1)
