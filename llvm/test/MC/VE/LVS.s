# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: lvs %s12, %v11(0)
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x0b,0x00,0x00,0x0c,0x9e]
lvs %s12, %v11(0)

# CHECK-INST: lvs %s63, %vix(%s23)
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0xff,0x00,0x97,0x3f,0x9e]
lvs %s63, %vix(%s23)

# CHECK-INST: lvs %s0, %v11(127)
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x0b,0x00,0x7f,0x00,0x9e]
lvs %s0, %v11(127)
