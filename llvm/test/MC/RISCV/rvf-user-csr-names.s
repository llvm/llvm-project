# RUN: llvm-mc %s -triple=riscv32 -riscv-no-aliases -mattr=+f -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+f < %s \
# RUN:     | llvm-objdump -d -mattr=+f - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+f < %s \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS-NO-F %s
#
# RUN: llvm-mc %s -triple=riscv64 -riscv-no-aliases -mattr=+f -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+f < %s \
# RUN:     | llvm-objdump -d -mattr=+f - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+f < %s \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS-NO-F %s

##################################
# User Floating Pont CSRs
##################################

# fflags
# name
# CHECK-INST: csrrs t1, fflags, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x10,0x00]
# CHECK-INST-ALIAS: frflags t1
# CHECK-INST-ALIAS-NO-F: csrr t1, 1
# uimm12
# CHECK-INST: csrrs t2, fflags, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x10,0x00]
# CHECK-INST-ALIAS: frflags t2
# CHECK-INST-ALIAS-NO-F: csrr t2, 1
# name
csrrs t1, fflags, zero
# uimm12
csrrs t2, 0x001, zero

# frm
# name
# CHECK-INST: csrrs t1, frm, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x20,0x00]
# CHECK-INST-ALIAS: frrm t1
# CHECK-INST-ALIAS-NO-F: csrr t1, 2
# uimm12
# CHECK-INST: csrrs t2, frm, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x20,0x00]
# CHECK-INST-ALIAS: frrm t2
# CHECK-INST-ALIAS-NO-F: csrr t2, 2
# name
csrrs t1, frm, zero
# uimm12
csrrs t2, 0x002, zero

# fcsr
# name
# CHECK-INST: csrrs t1, fcsr, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x30,0x00]
# CHECK-INST-ALIAS: frcsr t1
# CHECK-INST-ALIAS-NO-F: csrr t1, 3
# uimm12
# CHECK-INST: csrrs t2, fcsr, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x30,0x00]
# CHECK-INST-ALIAS: frcsr t2
# CHECK-INST-ALIAS-NO-F: csrr t2, 3
# name
csrrs t1, fcsr, zero
# uimm12
csrrs t2, 0x003, zero


