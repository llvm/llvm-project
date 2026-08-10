# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: pfch 8199
# CHECK-ENCODING: encoding: [0x07,0x20,0x00,0x00,0x00,0x00,0x00,0x0c]
pfch 8199

# CHECK-INST: pfch 20(%s11)
# CHECK-ENCODING: encoding: [0x14,0x00,0x00,0x00,0x00,0x8b,0x00,0x0c]
pfch 20(%s11)

# CHECK-INST: pfch -1(, %s11)
# CHECK-ENCODING: encoding: [0xff,0xff,0xff,0xff,0x8b,0x00,0x00,0x0c]
pfch -1(, %s11)

# CHECK-INST: pfch 20(%s10, %s11)
# CHECK-ENCODING: encoding: [0x14,0x00,0x00,0x00,0x8b,0x8a,0x00,0x0c]
pfch 20(%s10, %s11)
