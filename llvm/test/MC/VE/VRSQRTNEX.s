# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vrsqrt.d.nex %v11, %v22
# CHECK-ENCODING: encoding: [0x00,0x00,0x16,0x0b,0x00,0x00,0x10,0xf1]
vrsqrt.d.nex %v11, %v22

# CHECK-INST: pvrsqrt.up.nex %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0xff,0x00,0x00,0x90,0xf1]
vrsqrt.s.nex %vix, %vix

# CHECK-INST: pvrsqrt.lo.nex %vix, %v22
# CHECK-ENCODING: encoding: [0x00,0x00,0x16,0xff,0x00,0x00,0x50,0xf1]
pvrsqrt.lo.nex %vix, %v22

# CHECK-INST: pvrsqrt.up.nex %v11, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x00,0x16,0x0b,0x00,0x00,0x9b,0xf1]
pvrsqrt.up.nex %v11, %v22, %vm11

# CHECK-INST: pvrsqrt.up.nex %v11, %vix, %vm11
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0x0b,0x00,0x00,0x9b,0xf1]
pvrsqrt.up.nex %v11, %vix, %vm11

# CHECK-INST: pvrsqrt.nex %v12, %v20, %vm12
# CHECK-ENCODING: encoding: [0x00,0x00,0x14,0x0c,0x00,0x00,0xdc,0xf1]
pvrsqrt.nex %v12, %v20, %vm12
