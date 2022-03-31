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

################################
# Virtual Supervisor Registers
################################

# vstimecmph
# name
# CHECK-INST: csrrs t1, vstimecmph, zero
# CHECK-ENC: encoding: [0x73,0x23,0xd0,0x25]
# CHECK-INST-ALIAS: csrr t1, vstimecmph
# uimm12
# CHECK-INST: csrrs t2, vstimecmph, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xd0,0x25]
# CHECK-INST-ALIAS: csrr t2, vstimecmph
# name
csrrs t1, vstimecmph, zero
# uimm12
csrrs t2, 0x25D, zero

#########################################
# State Enable Extension (Smstateen)
#########################################

# hstateen0h
# name
# CHECK-INST: csrrs t1, hstateen0h, zero
# CHECK-ENC: encoding: [0x73,0x23,0xc0,0x61]
# CHECK-INST-ALIAS: csrr t1, hstateen0h
# uimm12
# CHECK-INST: csrrs t2, hstateen0h, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xc0,0x61]
# CHECK-INST-ALIAS: csrr t2, hstateen0h
# name
csrrs t1, hstateen0h, zero
# uimm12
csrrs t2, 0x61C, zero

# hstateen1h
# name
# CHECK-INST: csrrs t1, hstateen1h, zero
# CHECK-ENC: encoding: [0x73,0x23,0xd0,0x61]
# CHECK-INST-ALIAS: csrr t1, hstateen1h
# uimm12
# CHECK-INST: csrrs t2, hstateen1h, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xd0,0x61]
# CHECK-INST-ALIAS: csrr t2, hstateen1h
# name
csrrs t1, hstateen1h, zero
# uimm12
csrrs t2, 0x61D, zero

# hstateen2h
# name
# CHECK-INST: csrrs t1, hstateen2h, zero
# CHECK-ENC: encoding: [0x73,0x23,0xe0,0x61]
# CHECK-INST-ALIAS: csrr t1, hstateen2h
# uimm12
# CHECK-INST: csrrs t2, hstateen2h, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xe0,0x61]
# CHECK-INST-ALIAS: csrr t2, hstateen2h
# name
csrrs t1, hstateen2h, zero
# uimm12
csrrs t2, 0x61E, zero

# hstateen3h
# name
# CHECK-INST: csrrs t1, hstateen3h, zero
# CHECK-ENC: encoding: [0x73,0x23,0xf0,0x61]
# CHECK-INST-ALIAS: csrr t1, hstateen3h
# uimm12
# CHECK-INST: csrrs t2, hstateen3h, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xf0,0x61]
# CHECK-INST-ALIAS: csrr t2, hstateen3h
# name
csrrs t1, hstateen3h, zero
# uimm12
csrrs t2, 0x61F, zero
