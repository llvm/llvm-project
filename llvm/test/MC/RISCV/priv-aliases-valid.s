# RUN: llvm-mc %s -triple=riscv32 -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:     | llvm-objdump -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

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
