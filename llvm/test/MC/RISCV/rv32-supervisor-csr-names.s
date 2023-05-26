# RUN: llvm-mc %s -triple=riscv32 -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s

##################################
# Supervisor Trap Setup
##################################

# stimecmph
# name
# CHECK-INST: csrrs t1, stimecmph, zero
# CHECK-ENC: encoding: [0x73,0x23,0xd0,0x15]
# CHECK-INST-ALIAS: csrr t1, stimecmph
# uimm12
# CHECK-INST: csrrs t2, stimecmph, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xd0,0x15]
# CHECK-INST-ALIAS: csrr t2, stimecmph
# name
csrrs t1, stimecmph, zero
# uimm12
csrrs t2, 0x15D, zero

#########################################
# Advanced Interrupt Architecture (Smaia and Ssaia)
#########################################

# sieh
# name
# CHECK-INST: csrrs t1, sieh, zero
# CHECK-ENC: encoding: [0x73,0x23,0x40,0x11]
# CHECK-INST-ALIAS: csrr t1, sieh
# uimm12
# CHECK-INST: csrrs t2, sieh, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x40,0x11]
# CHECK-INST-ALIAS: csrr t2, sieh
# name
csrrs t1, sieh, zero
# uimm12
csrrs t2, 0x114, zero

# siph
# name
# CHECK-INST: csrrs t1, siph, zero
# CHECK-ENC: encoding: [0x73,0x23,0x40,0x15]
# CHECK-INST-ALIAS: csrr t1, siph
# uimm12
# CHECK-INST: csrrs t2, siph, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x40,0x15]
# CHECK-INST-ALIAS: csrr t2, siph
# name
csrrs t1, siph, zero
# uimm12
csrrs t2, 0x154, zero
