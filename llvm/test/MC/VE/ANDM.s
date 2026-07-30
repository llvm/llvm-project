# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: andm %vm0, %vm0, %vm0
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x84]
andm %vm0, %vm0, %vm0

# CHECK-INST: andm %vm11, %vm1, %vm15
# CHECK-ENCODING: encoding: [0x00,0x0f,0x01,0x0b,0x00,0x00,0x00,0x84]
andm %vm11, %vm1, %vm15

# CHECK-INST: andm %vm11, %vm15, %vm0
# CHECK-ENCODING: encoding: [0x00,0x00,0x0f,0x0b,0x00,0x00,0x00,0x84]
andm %vm11, %vm15, %vm0
