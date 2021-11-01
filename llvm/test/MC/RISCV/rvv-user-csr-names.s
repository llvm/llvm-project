# RUN: llvm-mc %s -triple=riscv32 -riscv-no-aliases -mattr=+f -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+v < %s \
# RUN:     | llvm-objdump -d --mattr=+v - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s
#
# RUN: llvm-mc %s -triple=riscv64 -riscv-no-aliases -mattr=+f -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+v < %s \
# RUN:     | llvm-objdump -d --mattr=+v - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s

##################################
# User Vector CSRs
##################################

# vstart
# name
# CHECK-INST: csrrs t1, vstart, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x80,0x00]
# CHECK-INST-ALIAS: csrr t1, vstart
# uimm12
# CHECK-INST: csrrs t2, vstart, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x80,0x00]
# CHECK-INST-ALIAS: csrr t2, vstart
# name
csrrs t1, vstart, zero
# uimm12
csrrs t2, 0x008, zero

# vxsat
# name
# CHECK-INST: csrrs t1, vxsat, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x90,0x00]
# CHECK-INST-ALIAS: csrr t1, vxsat
# uimm12
# CHECK-INST: csrrs t2, vxsat, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x90,0x00]
# CHECK-INST-ALIAS: csrr t2, vxsat
# name
csrrs t1, vxsat, zero
# uimm12
csrrs t2, 0x009, zero

# vxrm
# name
# CHECK-INST: csrrs t1, vxrm, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xa0,0x00]
# CHECK-INST-ALIAS: csrr t1, vxrm
# uimm12
# CHECK-INST: csrrs t2, vxrm, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xa0,0x00]
# CHECK-INST-ALIAS: csrr t2, vxrm
# name
csrrs t1, vxrm, zero
# uimm12
csrrs t2, 0x00a, zero

# vcsr
# name
# CHECK-INST: csrrs t1, vcsr, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xf0,0x00]
# CHECK-INST-ALIAS: csrr t1, vcsr
# uimm12
# CHECK-INST: csrrs t2, vcsr, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xf0,0x00]
# CHECK-INST-ALIAS: csrr t2, vcsr
# name
csrrs t1, vcsr, zero
# uimm12
csrrs t2, 0x00f, zero

# vl
# name
# CHECK-INST: csrrs t1, vl, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x00,0xc2]
# CHECK-INST-ALIAS: csrr t1, vl
# uimm12
# CHECK-INST: csrrs t2, vl, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x00,0xc2]
# CHECK-INST-ALIAS: csrr t2, vl
# name
csrrs t1, vl, zero
# uimm12
csrrs t2, 0xc20, zero

# vtype
# name
# CHECK-INST: csrrs t1, vtype, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x10,0xc2]
# CHECK-INST-ALIAS: csrr t1, vtype
# uimm12
# CHECK-INST: csrrs t2, vtype, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x10,0xc2]
# CHECK-INST-ALIAS: csrr t2, vtype
# name
csrrs t1, vtype, zero
# uimm12
csrrs t2, 0xc21, zero

# vlenb
# name
# CHECK-INST: csrrs t1, vlenb, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x20,0xc2]
# CHECK-INST-ALIAS: csrr t1, vlenb
# uimm12
# CHECK-INST: csrrs t2, vlenb, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x20,0xc2]
# CHECK-INST-ALIAS: csrr t2, vlenb
# name
csrrs t1, vlenb, zero
# uimm12
csrrs t2, 0xc22, zero
