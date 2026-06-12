# RUN: llvm-mc %s -triple=riscv32 -M no-aliases -mattr=+zcmt -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+zcmt < %s \
# RUN:     | llvm-objdump -d --mattr=+zcmt - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s
#
# RUN: llvm-mc %s -triple=riscv64 -M no-aliases -mattr=+zcmt -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+zcmt < %s \
# RUN:     | llvm-objdump -d --mattr=+zcmt - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s

##################################
# Jump Vector Table CSR
##################################

# jvt
# name
# CHECK-INST: csrrs t1, jvt, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x70,0x01]
# CHECK-INST-ALIAS: csrr t1, jvt
# uimm12
# CHECK-INST: csrrs t2, jvt, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x70,0x01]
# CHECK-INST-ALIAS: csrr t2, jvt
# name
csrrs t1, jvt, zero
# uimm12
csrrs t2, 0x017, zero
