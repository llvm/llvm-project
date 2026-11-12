# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: lhm.l %s20, 20(%s11)
# CHECK-ENCODING: encoding: [0x14,0x00,0x00,0x00,0x8b,0x03,0x14,0x21]
lhm.l %s20, 20(%s11)

# CHECK-INST: lhm.w %s20, 8192()
# CHECK-ENCODING: encoding: [0x00,0x20,0x00,0x00,0x00,0x02,0x14,0x21]
lhm.w %s20, 8192()

# CHECK-INST: lhm.h %s20, (%s11)
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x8b,0x01,0x14,0x21]
lhm.h %s20, (%s11)

# CHECK-INST: lhm.b %s20, (%s11)
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x8b,0x00,0x14,0x21]
lhm.b %s20, %s11
