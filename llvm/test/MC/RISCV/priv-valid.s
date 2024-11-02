# RUN: llvm-mc %s -triple=riscv32 -mattr=+svinval -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+svinval -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+svinval < %s \
# RUN:     | llvm-objdump --mattr=+svinval -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+svinval < %s \
# RUN:     | llvm-objdump --mattr=+svinval -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: uret
# CHECK: encoding: [0x73,0x00,0x20,0x00]
uret

# CHECK-INST: sret
# CHECK: encoding: [0x73,0x00,0x20,0x10]
sret

# CHECK-INST: mret
# CHECK: encoding: [0x73,0x00,0x20,0x30]
mret

# CHECK-INST: wfi
# CHECK: encoding: [0x73,0x00,0x50,0x10]
wfi

# CHECK-INST: sfence.vma zero, zero
# CHECK: encoding: [0x73,0x00,0x00,0x12]
sfence.vma zero, zero

# CHECK-INST: sfence.vma a0, a1
# CHECK: encoding: [0x73,0x00,0xb5,0x12]
sfence.vma a0, a1

# CHECK-INST: sinval.vma zero, zero
# CHECK: encoding: [0x73,0x00,0x00,0x16]
sinval.vma zero, zero

# CHECK-INST: sinval.vma a0, a1
# CHECK: encoding: [0x73,0x00,0xb5,0x16]
sinval.vma a0, a1

# CHECK-INST: sfence.w.inval
# CHECK: encoding: [0x73,0x00,0x00,0x18]
sfence.w.inval

# CHECK-INST: sfence.inval.ir
# CHECK: encoding: [0x73,0x00,0x10,0x18]
sfence.inval.ir

# CHECK-INST: hfence.vvma zero, zero
# CHECK: encoding: [0x73,0x00,0x00,0x22]
hfence.vvma zero, zero

# CHECK-INST: hfence.vvma a0, a1
# CHECK: encoding: [0x73,0x00,0xb5,0x22]
hfence.vvma a0, a1

# CHECK-INST: hfence.gvma zero, zero
# CHECK: encoding: [0x73,0x00,0x00,0x62]
hfence.gvma zero, zero

# CHECK-INST: hfence.gvma a0, a1
# CHECK: encoding: [0x73,0x00,0xb5,0x62]
hfence.gvma a0, a1

# CHECK-INST: hinval.vvma zero, zero
# CHECK: encoding: [0x73,0x00,0x00,0x26]
hinval.vvma zero, zero

# CHECK-INST: hinval.vvma a0, a1
# CHECK: encoding: [0x73,0x00,0xb5,0x26]
hinval.vvma a0, a1

# CHECK-INST: hinval.gvma zero, zero
# CHECK: encoding: [0x73,0x00,0x00,0x66]
hinval.gvma zero, zero

# CHECK-INST: hinval.gvma a0, a1
# CHECK: encoding: [0x73,0x00,0xb5,0x66]
hinval.gvma a0, a1

# CHECK-INST: hlv.b a0, (a1)
# CHECK: encoding: [0x73,0xc5,0x05,0x60]
hlv.b   a0, (a1)

# CHECK-INST: hlv.bu a0, (a1)
# CHECK: encoding: [0x73,0xc5,0x15,0x60]
hlv.bu  a0, (a1)

# CHECK-INST: hlv.h a1, (a2)
# CHECK: encoding: [0xf3,0x45,0x06,0x64]
hlv.h   a1, (a2)

# CHECK-INST: hlv.hu a1, (a1)
# CHECK: encoding: [0xf3,0xc5,0x15,0x64]
hlv.hu  a1, (a1)

# CHECK-INST: hlvx.hu a1, (a2)
# CHECK: encoding: [0xf3,0x45,0x36,0x64]
hlvx.hu a1, (a2)

# CHECK-INST: hlv.w a2, (a2)
# CHECK: encoding: [0x73,0x46,0x06,0x68]
hlv.w   a2, (a2)

# CHECK-INST: hlvx.wu a2, (a3)
# CHECK: encoding: [0x73,0xc6,0x36,0x68]
hlvx.wu a2, (a3)

# CHECK-INST: hsv.b a0, (a1)
# CHECK: encoding: [0x73,0xc0,0xa5,0x62]
hsv.b   a0, (a1)

# CHECK-INST: hsv.h a0, (a1)
# CHECK: encoding: [0x73,0xc0,0xa5,0x66]
hsv.h   a0, (a1)

# CHECK-INST: hsv.w a0, (a1)
# CHECK: encoding: [0x73,0xc0,0xa5,0x6a]
hsv.w   a0, (a1)
