# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vbrv %v11, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x00,0x00,0xf7]
vbrv %v11, %v22

# CHECK-INST: vbrv %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0xff,0x00,0xff,0x00,0x00,0x00,0xf7]
vbrv %vix, %vix

# CHECK-INST: pvbrv.lo %vix, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0xff,0x00,0x00,0x40,0xf7]
pvbrv.lo %vix, %v22

# CHECK-INST: pvbrv.lo %v11, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x00,0x4b,0xf7]
pvbrv.lo %v11, %v22, %vm11

# CHECK-INST: pvbrv.up %v11, %vix, %vm11
# CHECK-ENCODING: encoding: [0x00,0xff,0x00,0x0b,0x00,0x00,0x8b,0xf7]
pvbrv.up %v11, %vix, %vm11

# CHECK-INST: pvbrv %v12, %v20, %vm12
# CHECK-ENCODING: encoding: [0x00,0x14,0x00,0x0c,0x00,0x00,0xcc,0xf7]
pvbrv %v12, %v20, %vm12
