# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vmuls.w.sx %v11, %s20, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x94,0x20,0xcb]
vmuls.w.sx %v11, %s20, %v22

# CHECK-INST: vmuls.w.sx %vix, %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0xff,0xff,0xff,0x00,0x00,0x00,0xcb]
vmuls.w.sx %vix, %vix, %vix

# CHECK-INST: vmuls.w.zx %vix, 22, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0xff,0x00,0x16,0x60,0xcb]
vmuls.w.zx %vix, 22, %v22

# CHECK-INST: vmuls.w.zx %vix, 22, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0xff,0x00,0x16,0x60,0xcb]
vmuls.w %vix, 22, %v22

# CHECK-INST: vmuls.w.zx %v11, 63, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x3f,0x6b,0xcb]
vmuls.w %v11, 63, %v22, %vm11
