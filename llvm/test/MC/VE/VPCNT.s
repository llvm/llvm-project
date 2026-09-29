# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vpcnt %v11, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x00,0x00,0xac]
vpcnt %v11, %v22

# CHECK-INST: vpcnt %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0xff,0x00,0xff,0x00,0x00,0x00,0xac]
vpcnt %vix, %vix

# CHECK-INST: pvpcnt.lo %vix, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0xff,0x00,0x00,0x40,0xac]
pvpcnt.lo %vix, %v22

# CHECK-INST: pvpcnt.lo %v11, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x00,0x4b,0xac]
pvpcnt.lo %v11, %v22, %vm11

# CHECK-INST: pvpcnt.up %v11, %vix, %vm11
# CHECK-ENCODING: encoding: [0x00,0xff,0x00,0x0b,0x00,0x00,0x8b,0xac]
pvpcnt.up %v11, %vix, %vm11

# CHECK-INST: pvpcnt %v12, %v20, %vm12
# CHECK-ENCODING: encoding: [0x00,0x14,0x00,0x0c,0x00,0x00,0xcc,0xac]
pvpcnt %v12, %v20, %vm12
