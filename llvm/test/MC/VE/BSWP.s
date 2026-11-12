# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: bswp %s11, %s11, 0
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x8b,0x00,0x0b,0x2b]
bswp %s11, %s11, 0

# CHECK-INST: bswp %s11, %s11, 1
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x8b,0x01,0x0b,0x2b]
bswp %s11, %s11, 1

# CHECK-INST: bswp %s11, (32)1, 0
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x20,0x00,0x0b,0x2b]
bswp %s11, (32)1, 0

# CHECK-INST: bswp %s11, (32)0, 1
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x60,0x01,0x0b,0x2b]
bswp %s11, (32)0, 1
