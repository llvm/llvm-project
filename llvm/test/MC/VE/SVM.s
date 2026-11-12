# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: svm %s12, %vm11, 0
# CHECK-ENCODING: encoding: [0x00,0x0b,0x00,0x00,0x00,0x00,0x0c,0xa7]
svm %s12, %vm11, 0

# CHECK-INST: svm %s63, %vm0, %s23
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x00,0x97,0x3f,0xa7]
svm %s63, %vm0, %s23

# CHECK-INST: svm %s0, %vm11, 3
# CHECK-ENCODING: encoding: [0x00,0x0b,0x00,0x00,0x00,0x03,0x00,0xa7]
svm %s0, %vm11, 3
