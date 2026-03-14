# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: bsic %s11, 23
# CHECK-ENCODING: encoding: [0x17,0x00,0x00,0x00,0x00,0x00,0x0b,0x08]
bsic %s11, 23

# CHECK-INST: bsic %s63, 324(, %s11)
# CHECK-ENCODING: encoding: [0x44,0x01,0x00,0x00,0x8b,0x00,0x3f,0x08]
bsic %s63, 324(,%s11)

# CHECK-INST: bsic %s11, 324(%s10)
# CHECK-ENCODING: encoding: [0x44,0x01,0x00,0x00,0x00,0x8a,0x0b,0x08]
bsic %s11, 324(%s10  )

# CHECK-INST: bsic %s11, 324(%s13, %s11)
# CHECK-ENCODING: encoding: [0x44,0x01,0x00,0x00,0x8b,0x8d,0x0b,0x08]
bsic %s11, 324 (%s13,%s11)

# CHECK-INST: bsic %s11, (%s10)
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x00,0x8a,0x0b,0x08]
bsic %s11, (%s10)

# CHECK-INST: bsic %s11, (, %s12)
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x8c,0x00,0x0b,0x08]
bsic %s11, (,%s12)
