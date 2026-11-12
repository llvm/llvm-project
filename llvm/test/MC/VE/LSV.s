# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: lsv %v11(0), %s12
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x0b,0x8c,0x00,0x00,0x8e]
lsv %v11(0), %s12

# CHECK-INST: lsv %vix(%s23), %s12
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0xff,0x8c,0x97,0x00,0x8e]
lsv %vix(%s23), %s12

# CHECK-INST: lsv %v11(0), (32)0
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x0b,0x60,0x00,0x00,0x8e]
lsv %v11(0), (32)0

# CHECK-INST: lsv %vix(%s23), (23)1
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0xff,0x17,0x97,0x00,0x8e]
lsv %vix(%s23), (23)1

# CHECK-INST: lsv %v11(%s22), (1)0
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x0b,0x41,0x96,0x00,0x8e]
lsv %v11(%s22), (1)0

# CHECK-INST: lsv %vix(127), (63)1
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0xff,0x3f,0x7f,0x00,0x8e]
lsv %vix(127), (63)1
