# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vldz %v11, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x00,0x00,0xe7]
vldz %v11, %v22

# CHECK-INST: vldz %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0xff,0x00,0xff,0x00,0x00,0x00,0xe7]
vldz %vix, %vix

# CHECK-INST: pvldz.lo %vix, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0xff,0x00,0x00,0x40,0xe7]
pvldz.lo %vix, %v22

# CHECK-INST: pvldz.lo %v11, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x00,0x4b,0xe7]
pvldz.lo %v11, %v22, %vm11

# CHECK-INST: pvldz.up %v11, %vix, %vm11
# CHECK-ENCODING: encoding: [0x00,0xff,0x00,0x0b,0x00,0x00,0x8b,0xe7]
pvldz.up %v11, %vix, %vm11

# CHECK-INST: pvldz %v12, %v20, %vm12
# CHECK-ENCODING: encoding: [0x00,0x14,0x00,0x0c,0x00,0x00,0xcc,0xe7]
pvldz %v12, %v20, %vm12
