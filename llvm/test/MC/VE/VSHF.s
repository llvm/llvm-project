# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vshf %v11, %v20, %v22, %s20
# CHECK-ENCODING: encoding: [0x00,0x16,0x14,0x0b,0x00,0x94,0x00,0xbc]
vshf %v11, %v20, %v22, %s20

# CHECK-INST: vshf %vix, %vix, %vix, 0
# CHECK-ENCODING: encoding: [0x00,0xff,0xff,0xff,0x00,0x00,0x00,0xbc]
vshf %vix, %vix, %vix, 0

# CHECK-INST: vshf %vix, %vix, %v22, 15
# CHECK-ENCODING: encoding: [0x00,0x16,0xff,0xff,0x00,0x0f,0x00,0xbc]
vshf %vix, %vix, %v22, 15

# CHECK-INST: vshf %v11, %vix, %v22, 12
# CHECK-ENCODING: encoding: [0x00,0x16,0xff,0x0b,0x00,0x0c,0x00,0xbc]
vshf %v11, %vix, %v22, 12

# CHECK-INST: vshf %v11, %v23, %v22, %s63
# CHECK-ENCODING: encoding: [0x00,0x16,0x17,0x0b,0x00,0xbf,0x00,0xbc]
vshf %v11, %v23, %v22, %s63
