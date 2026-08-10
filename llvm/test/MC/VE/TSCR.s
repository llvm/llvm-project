# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: tscr %s11, %s20, %s22
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x96,0x94,0x0b,0x41]
tscr %s11, %s20, %s22

# CHECK-INST: tscr %s11, %s20, 0
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x00,0x94,0x0b,0x41]
tscr %s11, %s20, 0

# CHECK-INST: tscr %s11, 22, %s15
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x8f,0x16,0x0b,0x41]
tscr %s11, 22, %s15

# CHECK-INST: tscr %s11, 22, 0
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x00,0x16,0x0b,0x41]
tscr %s11, 22, 0
