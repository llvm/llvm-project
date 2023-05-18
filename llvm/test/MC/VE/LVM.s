# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: lvm %vm11, 0, %s12
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x0b,0x8c,0x00,0x00,0xb7]
lvm %vm11, 0, %s12

# CHECK-INST: lvm %vm12, %s23, %s12
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x0c,0x8c,0x97,0x00,0xb7]
lvm %vm12, %s23, %s12

# CHECK-INST: lvm %vm1, 0, (32)0
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x01,0x60,0x00,0x00,0xb7]
lvm %vm1, 0, (32)0

# CHECK-INST: lvm %vm2, %s23, (23)1
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x02,0x17,0x97,0x00,0xb7]
lvm %vm2, %s23, (23)1

# CHECK-INST: lvm %vm0, %s22, (1)0
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x41,0x96,0x00,0xb7]
lvm %vm0, %s22, (1)0

# CHECK-INST: lvm %vm15, 3, (63)1
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x0f,0x3f,0x03,0x00,0xb7]
lvm %vm15, 3, (63)1
