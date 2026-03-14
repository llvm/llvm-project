# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vrsqrt.d %v11, %v22
# CHECK-ENCODING: encoding: [0x00,0x00,0x16,0x0b,0x00,0x00,0x00,0xf1]
vrsqrt.d %v11, %v22

# CHECK-INST: pvrsqrt.up %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0xff,0x00,0x00,0x80,0xf1]
vrsqrt.s %vix, %vix

# CHECK-INST: pvrsqrt.lo %vix, %v22
# CHECK-ENCODING: encoding: [0x00,0x00,0x16,0xff,0x00,0x00,0x40,0xf1]
pvrsqrt.lo %vix, %v22

# CHECK-INST: pvrsqrt.up %v11, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x00,0x16,0x0b,0x00,0x00,0x8b,0xf1]
pvrsqrt.up %v11, %v22, %vm11

# CHECK-INST: pvrsqrt.up %v11, %vix, %vm11
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0x0b,0x00,0x00,0x8b,0xf1]
pvrsqrt.up %v11, %vix, %vm11

# CHECK-INST: pvrsqrt %v12, %v20, %vm12
# CHECK-ENCODING: encoding: [0x00,0x00,0x14,0x0c,0x00,0x00,0xcc,0xf1]
pvrsqrt %v12, %v20, %vm12
