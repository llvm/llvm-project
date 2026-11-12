# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vex %v11, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x00,0x00,0x9d]
vex %v11, %v22

# CHECK-INST: vex %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0xff,0x00,0xff,0x00,0x00,0x00,0x9d]
vex %vix, %vix

# CHECK-INST: vex %vix, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0xff,0x00,0x00,0x00,0x9d]
vex %vix, %v22

# CHECK-INST: vex %v11, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x00,0x0b,0x9d]
vex %v11, %v22, %vm11

# CHECK-INST: vex %v11, %vix, %vm11
# CHECK-ENCODING: encoding: [0x00,0xff,0x00,0x0b,0x00,0x00,0x0b,0x9d]
vex %v11, %vix, %vm11

# CHECK-INST: vex %v12, %v20, %vm12
# CHECK-ENCODING: encoding: [0x00,0x14,0x00,0x0c,0x00,0x00,0x0c,0x9d]
vex %v12, %v20, %vm12
