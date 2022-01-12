# RUN: llvm-mc %s -triple=riscv32 -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s

##################################
# Hypervisor Configuration
##################################

# henvcfgh
# name
# CHECK-INST: csrrs t1, henvcfgh, zero
# CHECK-ENC: encoding: [0x73,0x23,0xa0,0x61]
# CHECK-INST-ALIAS: csrr t1, henvcfgh
# uimm12
# CHECK-INST: csrrs t2, henvcfgh, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xa0,0x61]
# CHECK-INST-ALIAS: csrr t2, henvcfgh
# name
csrrs t1, henvcfgh, zero
# uimm12
csrrs t2, 0x61A, zero

#####################################################
# Hypervisor Counter/Timer Virtualization Registers
#####################################################

# htimedeltah
# name
# CHECK-INST: csrrs t1, htimedeltah, zero
# CHECK-ENC: encoding: [0x73,0x23,0x50,0x61]
# CHECK-INST-ALIAS: csrr t1, htimedeltah
# uimm12
# CHECK-INST: csrrs t2, htimedeltah, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x50,0x61]
# CHECK-INST-ALIAS: csrr t2, htimedeltah
# name
csrrs t1, htimedeltah, zero
# uimm12
csrrs t2, 0x615, zero
