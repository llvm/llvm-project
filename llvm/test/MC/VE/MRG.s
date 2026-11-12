# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: mrg %s11, %s11, %s11
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x8b,0x8b,0x0b,0x56]
mrg %s11, %s11, %s11

# CHECK-INST: mrg %s11, 63, %s11
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x8b,0x3f,0x0b,0x56]
mrg %s11, 63, %s11

# CHECK-INST: mrg %s11, -1, %s11
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x8b,0x7f,0x0b,0x56]
mrg %s11, -1, %s11

# CHECK-INST: mrg %s11, -64, %s11
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x8b,0x40,0x0b,0x56]
mrg %s11, -64, %s11

# CHECK-INST: mrg %s11, -64, (32)1
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x20,0x40,0x0b,0x56]
mrg %s11, -64, (32)1

# CHECK-INST: mrg %s11, 63, (32)0
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x60,0x3f,0x0b,0x56]
mrg %s11, 63, (32)0
