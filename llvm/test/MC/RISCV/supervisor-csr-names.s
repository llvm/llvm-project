# RUN: llvm-mc %s -triple=riscv32 -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s
#
# RUN: llvm-mc %s -triple=riscv64 -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s

##################################
# Supervisor Trap Setup
##################################

# sstatus
# name
# CHECK-INST: csrrs t1, sstatus, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0x10]
# CHECK-INST-ALIAS: csrr t1, sstatus
# uimm12
# CHECK-INST: csrrs t2, sstatus, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0x10]
# CHECK-INST-ALIAS: csrr t2, sstatus
# name
csrrs t1, sstatus, zero
# uimm12
csrrs t2, 0x100, zero

# sie
# name
# CHECK-INST: csrrs t1, sie, zero
# CHECK-ENC: [0x73,0x23,0x40,0x10]
# CHECK-INST-ALIAS: csrr t1, sie
# uimm12
# CHECK-INST: csrrs t2, sie, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x40,0x10]
# CHECK-INST-ALIAS: csrr t2, sie
# name
csrrs t1, sie, zero
# uimm12
csrrs t2, 0x104, zero

# stvec
# name
# CHECK-INST: csrrs t1, stvec, zero
# CHECK-ENC: encoding: [0x73,0x23,0x50,0x10]
# CHECK-INST-ALIAS: csrr t1, stvec
# uimm12
# CHECK-INST: csrrs t2, stvec, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x50,0x10]
# CHECK-INST-ALIAS: csrr t2, stvec
# name
csrrs t1, stvec, zero
# uimm12
csrrs t2, 0x105, zero

# scounteren
# name
# CHECK-INST: csrrs t1, scounteren, zero
# CHECK-ENC: encoding: [0x73,0x23,0x60,0x10]
# CHECK-INST-ALIAS: csrr t1, scounteren
# uimm12
# CHECK-INST: csrrs t2, scounteren, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x60,0x10]
# CHECK-INST-ALIAS: csrr t2, scounteren
# name
csrrs t1, scounteren, zero
# uimm12
csrrs t2, 0x106, zero

# stimecmp
# name
# CHECK-INST: csrrs t1, stimecmp, zero
# CHECK-ENC: encoding: [0x73,0x23,0xd0,0x14]
# CHECK-INST-ALIAS: csrr t1, stimecmp
# uimm12
# CHECK-INST: csrrs t2, stimecmp, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xd0,0x14]
# CHECK-INST-ALIAS: csrr t2, stimecmp
# name
csrrs t1, stimecmp, zero
# uimm12
csrrs t2, 0x14D, zero

##################################
# Supervisor Configuration
##################################

# senvcfg
# name
# CHECK-INST: csrrs t1, senvcfg, zero
# CHECK-ENC: encoding: [0x73,0x23,0xa0,0x10]
# CHECK-INST-ALIAS: csrr t1, senvcfg
# uimm12
# CHECK-INST: csrrs t2, senvcfg, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xa0,0x10]
# CHECK-INST-ALIAS: csrr t2, senvcfg
# name
csrrs t1, senvcfg, zero
# uimm12
csrrs t2, 0x10A, zero

##################################
# Supervisor Trap Handling
##################################

# sscratch
# name
# CHECK-INST: csrrs t1, sscratch, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0x14]
# CHECK-INST-ALIAS: csrr t1, sscratch
# uimm12
# CHECK-INST: csrrs t2, sscratch, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0x14]
# CHECK-INST-ALIAS: csrr t2, sscratch
# name
csrrs t1, sscratch, zero
# uimm12
csrrs t2, 0x140, zero

# sepc
# name
# CHECK-INST: csrrs t1, sepc, zero
# CHECK-ENC: encoding: [0x73,0x23,0x10,0x14]
# CHECK-INST-ALIAS: csrr t1, sepc
# uimm12
# CHECK-INST: csrrs t2, sepc, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x10,0x14]
# CHECK-INST-ALIAS: csrr t2, sepc
# name
csrrs t1, sepc, zero
# uimm12
csrrs t2, 0x141, zero

# scause
# name
# CHECK-INST: csrrs t1, scause, zero
# CHECK-ENC: encoding: [0x73,0x23,0x20,0x14]
# CHECK-INST-ALIAS: csrr t1, scause
# uimm12
# CHECK-INST: csrrs t2, scause, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x20,0x14]
# CHECK-INST-ALIAS: csrr t2, scause
# name
csrrs t1, scause, zero
# uimm12
csrrs t2, 0x142, zero

# stval
# name
# CHECK-INST: csrrs t1, stval, zero
# CHECK-ENC: encoding: [0x73,0x23,0x30,0x14]
# CHECK-INST-ALIAS: csrr t1, stval
# uimm12
# CHECK-INST: csrrs t2, stval, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x30,0x14]
# CHECK-INST-ALIAS: csrr t2, stval
# aliases
# aliases with uimm12
# name
csrrs t1, stval, zero
# uimm12
csrrs t2, 0x143, zero

# sip
# name
# CHECK-INST: csrrs t1, sip, zero
# CHECK-ENC: encoding: [0x73,0x23,0x40,0x14]
# CHECK-INST-ALIAS: csrr t1, sip
# uimm12
# CHECK-INST: csrrs t2, sip, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x40,0x14]
# CHECK-INST-ALIAS: csrr t2, sip
csrrs t1, sip, zero
# uimm12
csrrs t2, 0x144, zero


#########################################
# Supervisor Protection and Translation
#########################################

# satp
# name
# CHECK-INST: csrrs t1, satp, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0x18]
# CHECK-INST-ALIAS: csrr t1, satp
# uimm12
# CHECK-INST: csrrs t2, satp, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0x18]
# CHECK-INST-ALIAS: csrr t2, satp
# name
csrrs t1, satp, zero
# uimm12
csrrs t2, 0x180, zero

#########################################
# Quality-of-Service(QoS) Identifiers
#########################################

# srmcfg
# name
# CHECK-INST: csrrs t1, srmcfg, zero
# CHECK-ENC: encoding: [0x73,0x23,0x10,0x18]
# CHECK-INST-ALIAS: csrr t1, srmcfg
# uimm12
# CHECK-INST: csrrs t2, srmcfg, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x10,0x18]
# CHECK-INST-ALIAS: csrr t2, srmcfg
# name
csrrs t1, srmcfg, zero
# uimm12
csrrs t2, 0x181, zero

#########################################
# Debug/Trace Registers
#########################################

# scontext
# name
# CHECK-INST: csrrs t1, scontext, zero
# CHECK-ENC: encoding: [0x73,0x23,0x80,0x5a]
# CHECK-INST-ALIAS: csrr t1, scontext
# uimm12
# CHECK-INST: csrrs t2, scontext, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x80,0x5a]
# CHECK-INST-ALIAS: csrr t2, scontext
# name
csrrs t1, scontext, zero
# uimm12
csrrs t2, 0x5A8, zero

#########################################
# Supervisor Count Overflow (Sscofpmf)
#########################################

# scountovf
# name
# CHECK-INST: csrrs t1, scountovf, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0xda]
# CHECK-INST-ALIAS: csrr t1, scountovf
# uimm12
# CHECK-INST: csrrs t2, scountovf, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0xda]
# CHECK-INST-ALIAS: csrr t2, scountovf
# name
csrrs t1, scountovf, zero
# uimm12
csrrs t2, 0xDA0, zero

#########################################
# State Enable Extension (Smstateen)
#########################################

# sstateen0
# name
# CHECK-INST: csrrs t1, sstateen0, zero
# CHECK-ENC: encoding: [0x73,0x23,0xc0,0x10]
# CHECK-INST-ALIAS: csrr t1, sstateen0
# uimm12
# CHECK-INST: csrrs t2, sstateen0, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xc0,0x10]
# CHECK-INST-ALIAS: csrr t2, sstateen0
# name
csrrs t1, sstateen0, zero
# uimm12
csrrs t2, 0x10C, zero

# sstateen1
# name
# CHECK-INST: csrrs t1, sstateen1, zero
# CHECK-ENC: encoding: [0x73,0x23,0xd0,0x10]
# CHECK-INST-ALIAS: csrr t1, sstateen1
# uimm12
# CHECK-INST: csrrs t2, sstateen1, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xd0,0x10]
# CHECK-INST-ALIAS: csrr t2, sstateen1
# name
csrrs t1, sstateen1, zero
# uimm12
csrrs t2, 0x10D, zero

# sstateen2
# name
# CHECK-INST: csrrs t1, sstateen2, zero
# CHECK-ENC: encoding: [0x73,0x23,0xe0,0x10]
# CHECK-INST-ALIAS: csrr t1, sstateen2
# uimm12
# CHECK-INST: csrrs t2, sstateen2, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xe0,0x10]
# CHECK-INST-ALIAS: csrr t2, sstateen2
# name
csrrs t1, sstateen2, zero
# uimm12
csrrs t2, 0x10E, zero

# sstateen3
# name
# CHECK-INST: csrrs t1, sstateen3, zero
# CHECK-ENC: encoding: [0x73,0x23,0xf0,0x10]
# CHECK-INST-ALIAS: csrr t1, sstateen3
# uimm12
# CHECK-INST: csrrs t2, sstateen3, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xf0,0x10]
# CHECK-INST-ALIAS: csrr t2, sstateen3
# name
csrrs t1, sstateen3, zero
# uimm12
csrrs t2, 0x10F, zero

#########################################
# Advanced Interrupt Architecture (Smaia and Ssaia)
#########################################

# siselect
# name
# CHECK-INST: csrrs t1, siselect, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0x15]
# CHECK-INST-ALIAS: csrr t1, siselect
# uimm12
# CHECK-INST: csrrs t2, siselect, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0x15]
# CHECK-INST-ALIAS: csrr t2, siselect
# name
csrrs t1, siselect, zero
# uimm12
csrrs t2, 0x150, zero

# sireg
# name
# CHECK-INST: csrrs t1, sireg, zero
# CHECK-ENC: encoding: [0x73,0x23,0x10,0x15]
# CHECK-INST-ALIAS: csrr t1, sireg
# uimm12
# CHECK-INST: csrrs t2, sireg, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x10,0x15]
# CHECK-INST-ALIAS: csrr t2, sireg
# name
csrrs t1, sireg, zero
# uimm12
csrrs t2, 0x151, zero

# sireg2
# name
# CHECK-INST: csrrs t1, sireg2, zero
# CHECK-ENC: encoding: [0x73,0x23,0x20,0x15]
# CHECK-INST-ALIAS: csrr t1, sireg2
# uimm12
# CHECK-INST: csrrs t2, sireg2, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x20,0x15]
# CHECK-INST-ALIAS: csrr t2, sireg2
# name
csrrs t1, sireg2, zero
# uimm12
csrrs t2, 0x152, zero

# sireg3
# name
# CHECK-INST: csrrs t1, sireg3, zero
# CHECK-ENC: encoding: [0x73,0x23,0x30,0x15]
# CHECK-INST-ALIAS: csrr t1, sireg3
# uimm12
# CHECK-INST: csrrs t2, sireg3, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x30,0x15]
# CHECK-INST-ALIAS: csrr t2, sireg3
# name
csrrs t1, sireg3, zero
# uimm12
csrrs t2, 0x153, zero

# sireg4
# name
# CHECK-INST: csrrs t1, sireg4, zero
# CHECK-ENC: encoding: [0x73,0x23,0x50,0x15]
# CHECK-INST-ALIAS: csrr t1, sireg4
# uimm12
# CHECK-INST: csrrs t2, sireg4, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x50,0x15]
# CHECK-INST-ALIAS: csrr t2, sireg4
# name
csrrs t1, sireg4, zero
# uimm12
csrrs t2, 0x155, zero

# sireg5
# name
# CHECK-INST: csrrs t1, sireg5, zero
# CHECK-ENC: encoding: [0x73,0x23,0x60,0x15]
# CHECK-INST-ALIAS: csrr t1, sireg5
# uimm12
# CHECK-INST: csrrs t2, sireg5, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x60,0x15]
# CHECK-INST-ALIAS: csrr t2, sireg5
# name
csrrs t1, sireg5, zero
# uimm12
csrrs t2, 0x156, zero

# sireg6
# name
# CHECK-INST: csrrs t1, sireg6, zero
# CHECK-ENC: encoding: [0x73,0x23,0x70,0x15]
# CHECK-INST-ALIAS: csrr t1, sireg6
# uimm12
# CHECK-INST: csrrs t2, sireg6, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x70,0x15]
# CHECK-INST-ALIAS: csrr t2, sireg6
# name
csrrs t1, sireg6, zero
# uimm12
csrrs t2, 0x157, zero

# stopei
# name
# CHECK-INST: csrrs t1, stopei, zero
# CHECK-ENC: encoding: [0x73,0x23,0xc0,0x15]
# CHECK-INST-ALIAS: csrr t1, stopei
# uimm12
# CHECK-INST: csrrs t2, stopei, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xc0,0x15]
# CHECK-INST-ALIAS: csrr t2, stopei
# name
csrrs t1, stopei, zero
# uimm12
csrrs t2, 0x15C, zero

# stopi
# name
# CHECK-INST: csrrs t1, stopi, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0xdb]
# CHECK-INST-ALIAS: csrr t1, stopi
# uimm12
# CHECK-INST: csrrs t2, stopi, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0xdb]
# CHECK-INST-ALIAS: csrr t2, stopi
# name
csrrs t1, stopi, zero
# uimm12
csrrs t2, 0xDB0, zero

#########################################
# Counter Configuration (Ssccfg)
#########################################

# scountinhibit
# name
# CHECK-INST: csrrs t1, scountinhibit, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0x12]
# CHECK-INST-ALIAS: csrr t1, scountinhibit
# uimm12
# CHECK-INST: csrrs t2, scountinhibit, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0x12]
# CHECK-INST-ALIAS: csrr t2, scountinhibit
# name
csrrs t1, scountinhibit, zero
# uimm12
csrrs t2, 0x120, zero
