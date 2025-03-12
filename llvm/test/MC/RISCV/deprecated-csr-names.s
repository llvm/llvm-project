# RUN: llvm-mc -triple riscv32 -M no-aliases -show-encoding %s \
# RUN:     | FileCheck -check-prefixes CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype obj -triple riscv32 %s \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s

# RUN: llvm-mc -triple riscv64 -M no-aliases -show-encoding %s \
# RUN:     | FileCheck -check-prefixes CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype obj -triple riscv64 %s \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s

# RUN: llvm-mc -triple riscv32 %s 2>&1 | FileCheck -check-prefix CHECK-WARN %s

# sbadaddr
# name
# CHECK-INST: csrrw zero, stval, zero
# CHECK-ENC: encoding: [0x73,0x10,0x30,0x14]
# CHECK-INST-ALIAS: csrw stval, zero
# uimm12
# CHECK-INST: csrrw zero, stval, zero
# CHECK-ENC: encoding: [0x73,0x10,0x30,0x14]
# CHECK-INST-ALIAS: csrw stval, zero
# name
csrw sbadaddr, zero
# uimm12
csrrw zero, 0x143, zero

# CHECK-WARN: warning: 'sbadaddr' is a deprecated alias for 'stval'

# mbadaddr
# name
# CHECK-INST: csrrw zero, mtval, zero
# CHECK-ENC: encoding: [0x73,0x10,0x30,0x34]
# CHECK-INST-ALIAS: csrw mtval, zero
# uimm12
# CHECK-INST: csrrw zero, mtval, zero
# CHECK-ENC: encoding: [0x73,0x10,0x30,0x34]
# CHECK-INST-ALIAS: csrw mtval, zero
# name
csrw mbadaddr, zero
# uimm12
csrrw zero, 0x343, zero

# CHECK-WARN: warning: 'mbadaddr' is a deprecated alias for 'mtval'

# sptbr
# name
# CHECK-INST: csrrw zero, satp, zero
# CHECK-ENC: encoding: [0x73,0x10,0x00,0x18]
# CHECK-INST-ALIAS: csrw satp, zero
# uimm12
# CHECK-INST: csrrw zero, satp, zero
# CHECK-ENC: encoding: [0x73,0x10,0x00,0x18]
# CHECK-INST-ALIAS: csrw satp, zero
# name
csrw sptbr, zero
# uimm12
csrrw zero, 0x180, zero

# CHECK-WARN: warning: 'sptbr' is a deprecated alias for 'satp'
