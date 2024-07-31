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
# Hypervisor Trap Setup
##################################

# hstatus
# name
# CHECK-INST: csrrs t1, hstatus, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0x60]
# CHECK-INST-ALIAS: csrr t1, hstatus
# uimm12
# CHECK-INST: csrrs t2, hstatus, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0x60]
# CHECK-INST-ALIAS: csrr t2, hstatus
# name
csrrs t1, hstatus, zero
# uimm12
csrrs t2, 0x600, zero

# hedeleg
# name
# CHECK-INST: csrrs t1, hedeleg, zero
# CHECK-ENC: encoding: [0x73,0x23,0x20,0x60]
# CHECK-INST-ALIAS: csrr t1, hedeleg
# uimm12
# CHECK-INST: csrrs t2, hedeleg, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x20,0x60]
# CHECK-INST-ALIAS: csrr t2, hedeleg
# name
csrrs t1, hedeleg, zero
# uimm12
csrrs t2, 0x602, zero

# hideleg
# name
# CHECK-INST: csrrs t1, hideleg, zero
# CHECK-ENC: encoding: [0x73,0x23,0x30,0x60]
# CHECK-INST-ALIAS: csrr t1, hideleg
# uimm12
# CHECK-INST: csrrs t2, hideleg, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x30,0x60]
# CHECK-INST-ALIAS: csrr t2, hideleg
# name
csrrs t1, hideleg, zero
# uimm12
csrrs t2, 0x603, zero

# hie
# name
# CHECK-INST: csrrs t1, hie, zero
# CHECK-ENC: encoding: [0x73,0x23,0x40,0x60]
# CHECK-INST-ALIAS: csrr t1, hie
# uimm12
# CHECK-INST: csrrs t2, hie, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x40,0x60]
# CHECK-INST-ALIAS: csrr t2, hie
# name
csrrs t1, hie, zero
# uimm12
csrrs t2, 0x604, zero

# hcounteren
# name
# CHECK-INST: csrrs t1, hcounteren, zero
# CHECK-ENC: encoding: [0x73,0x23,0x60,0x60]
# CHECK-INST-ALIAS: csrr t1, hcounteren
# uimm12
# CHECK-INST: csrrs t2, hcounteren, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x60,0x60]
# CHECK-INST-ALIAS: csrr t2, hcounteren
# name
csrrs t1, hcounteren, zero
# uimm12
csrrs t2, 0x606, zero

# hgeie
# name
# CHECK-INST: csrrs t1, hgeie, zero
# CHECK-ENC: encoding: [0x73,0x23,0x70,0x60]
# CHECK-INST-ALIAS: csrr t1, hgeie
# uimm12
# CHECK-INST: csrrs t2, hgeie, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x70,0x60]
# CHECK-INST-ALIAS: csrr t2, hgeie
# name
csrrs t1, hgeie, zero
# uimm12
csrrs t2, 0x607, zero

##################################
# Hypervisor Trap Handling
##################################

# htval
# name
# CHECK-INST: csrrs t1, htval, zero
# CHECK-ENC: encoding: [0x73,0x23,0x30,0x64]
# CHECK-INST-ALIAS: csrr t1, htval
# uimm12
# CHECK-INST: csrrs t2, htval, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x30,0x64]
# CHECK-INST-ALIAS: csrr t2, htval
# name
csrrs t1, htval, zero
# uimm12
csrrs t2, 0x643, zero

# hip
# name
# CHECK-INST: csrrs t1, hip, zero
# CHECK-ENC: encoding: [0x73,0x23,0x40,0x64]
# CHECK-INST-ALIAS: csrr t1, hip
# uimm12
# CHECK-INST: csrrs t2, hip, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x40,0x64]
# CHECK-INST-ALIAS: csrr t2, hip
# name
csrrs t1, hip, zero
# uimm12
csrrs t2, 0x644, zero

# hvip
# name
# CHECK-INST: csrrs t1, hvip, zero
# CHECK-ENC: encoding: [0x73,0x23,0x50,0x64]
# CHECK-INST-ALIAS: csrr t1, hvip
# uimm12
# CHECK-INST: csrrs t2, hvip, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x50,0x64]
# CHECK-INST-ALIAS: csrr t2, hvip
# name
csrrs t1, hvip, zero
# uimm12
csrrs t2, 0x645, zero

# htinst
# name
# CHECK-INST: csrrs t1, htinst, zero
# CHECK-ENC: encoding: [0x73,0x23,0xa0,0x64]
# CHECK-INST-ALIAS: csrr t1, htinst
# uimm12
# CHECK-INST: csrrs t2, htinst, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xa0,0x64]
# CHECK-INST-ALIAS: csrr t2, htinst
# name
csrrs t1, htinst, zero
# uimm12
csrrs t2, 0x64A, zero

# hgeip
# name
# CHECK-INST: csrrs t1, hgeip, zero
# CHECK-ENC: encoding: [0x73,0x23,0x20,0xe1]
# CHECK-INST-ALIAS: csrr t1, hgeip
# uimm12
# CHECK-INST: csrrs t2, hgeip, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x20,0xe1]
# CHECK-INST-ALIAS: csrr t2, hgeip
# name
csrrs t1, hgeip, zero
# uimm12
csrrs t2, 0xE12, zero

##################################
# Hypervisor Configuration
##################################

# henvcfg
# name
# CHECK-INST: csrrs t1, henvcfg, zero
# CHECK-ENC: encoding: [0x73,0x23,0xa0,0x60]
# CHECK-INST-ALIAS: csrr t1, henvcfg
# uimm12
# CHECK-INST: csrrs t2, henvcfg, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xa0,0x60]
# CHECK-INST-ALIAS: csrr t2, henvcfg
# name
csrrs t1, henvcfg, zero
# uimm12
csrrs t2, 0x60A, zero

########################################
# Hypervisor Protection and Translation
########################################

# hgatp
# name
# CHECK-INST: csrrs t1, hgatp, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0x68]
# CHECK-INST-ALIAS: csrr t1, hgatp
# uimm12
# CHECK-INST: csrrs t2, hgatp, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0x68]
# CHECK-INST-ALIAS: csrr t2, hgatp
# name
csrrs t1, hgatp, zero
# uimm12
csrrs t2, 0x680, zero

##########################
# Debug/Trace Registers
##########################

# hcontext
# name
# CHECK-INST: csrrs t1, hcontext, zero
# CHECK-ENC: encoding: [0x73,0x23,0x80,0x6a]
# CHECK-INST-ALIAS: csrr t1, hcontext
# uimm12
# CHECK-INST: csrrs t2, hcontext, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x80,0x6a]
# CHECK-INST-ALIAS: csrr t2, hcontext
# name
csrrs t1, hcontext, zero
# uimm12
csrrs t2, 0x6A8, zero

####################################################
# Hypervisor Counter/Timer Virtualization Registers
####################################################

# htimedelta
# name
# CHECK-INST: csrrs t1, htimedelta, zero
# CHECK-ENC: encoding: [0x73,0x23,0x50,0x60]
# CHECK-INST-ALIAS: csrr t1, htimedelta
# uimm12
# CHECK-INST: csrrs t2, htimedelta, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x50,0x60]
# CHECK-INST-ALIAS: csrr t2, htimedelta
# name
csrrs t1, htimedelta, zero
# uimm12
csrrs t2, 0x605, zero

################################
# Virtual Supervisor Registers
################################

# vsstatus
# name
# CHECK-INST: csrrs t1, vsstatus, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0x20]
# CHECK-INST-ALIAS: csrr t1, vsstatus
# uimm12
# CHECK-INST: csrrs t2, vsstatus, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0x20]
# CHECK-INST-ALIAS: csrr t2, vsstatus
# name
csrrs t1, vsstatus, zero
# uimm12
csrrs t2, 0x200, zero

# vsie
# name
# CHECK-INST: csrrs t1, vsie, zero
# CHECK-ENC: encoding: [0x73,0x23,0x40,0x20]
# CHECK-INST-ALIAS: csrr t1, vsie
# uimm12
# CHECK-INST: csrrs t2, vsie, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x40,0x20]
# CHECK-INST-ALIAS: csrr t2, vsie
# name
csrrs t1, vsie, zero
# uimm12
csrrs t2, 0x204, zero

# vstvec
# name
# CHECK-INST: csrrs t1, vstvec, zero
# CHECK-ENC: encoding: [0x73,0x23,0x50,0x20]
# CHECK-INST-ALIAS: csrr t1, vstvec
# uimm12
# CHECK-INST: csrrs t2, vstvec, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x50,0x20]
# CHECK-INST-ALIAS: csrr t2, vstvec
# name
csrrs t1, vstvec, zero
# uimm12
csrrs t2, 0x205, zero

# vsscratch
# name
# CHECK-INST: csrrs t1, vsscratch, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0x24]
# CHECK-INST-ALIAS: csrr t1, vsscratch
# uimm12
# CHECK-INST: csrrs t2, vsscratch, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0x24]
# CHECK-INST-ALIAS: csrr t2, vsscratch
# name
csrrs t1, vsscratch, zero
# uimm12
csrrs t2, 0x240, zero

# vsepc
# name
# CHECK-INST: csrrs t1, vsepc, zero
# CHECK-ENC: encoding: [0x73,0x23,0x10,0x24]
# CHECK-INST-ALIAS: csrr t1, vsepc
# uimm12
# CHECK-INST: csrrs t2, vsepc, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x10,0x24]
# CHECK-INST-ALIAS: csrr t2, vsepc
# name
csrrs t1, vsepc, zero
# uimm12
csrrs t2, 0x241, zero

# vscause
# name
# CHECK-INST: csrrs t1, vscause, zero
# CHECK-ENC: encoding: [0x73,0x23,0x20,0x24]
# CHECK-INST-ALIAS: csrr t1, vscause
# uimm12
# CHECK-INST: csrrs t2, vscause, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x20,0x24]
# CHECK-INST-ALIAS: csrr t2, vscause
# name
csrrs t1, vscause, zero
# uimm12
csrrs t2, 0x242, zero

# vstval
# name
# CHECK-INST: csrrs t1, vstval, zero
# CHECK-ENC: encoding: [0x73,0x23,0x30,0x24]
# CHECK-INST-ALIAS: csrr t1, vstval
# uimm12
# CHECK-INST: csrrs t2, vstval, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x30,0x24]
# CHECK-INST-ALIAS: csrr t2, vstval
# name
csrrs t1, vstval, zero
# uimm12
csrrs t2, 0x243, zero

# vsip
# name
# CHECK-INST: csrrs t1, vsip, zero
# CHECK-ENC: encoding: [0x73,0x23,0x40,0x24]
# CHECK-INST-ALIAS: csrr t1, vsip
# uimm12
# CHECK-INST: csrrs t2, vsip, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x40,0x24]
# CHECK-INST-ALIAS: csrr t2, vsip
# name
csrrs t1, vsip, zero
# uimm12
csrrs t2, 0x244, zero

# vstimecmp
# name
# CHECK-INST: csrrs t1, vstimecmp, zero
# CHECK-ENC: encoding: [0x73,0x23,0xd0,0x24]
# CHECK-INST-ALIAS: csrr t1, vstimecmp
# uimm12
# CHECK-INST: csrrs t2, vstimecmp, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xd0,0x24]
# CHECK-INST-ALIAS: csrr t2, vstimecmp
# name
csrrs t1, vstimecmp, zero
# uimm12
csrrs t2, 0x24D, zero

# vsatp
# name
# CHECK-INST: csrrs t1, vsatp, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0x28]
# CHECK-INST-ALIAS: csrr t1, vsatp
# uimm12
# CHECK-INST: csrrs t2, vsatp, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0x28]
# CHECK-INST-ALIAS: csrr t2, vsatp
# name
csrrs t1, vsatp, zero
# uimm12
csrrs t2, 0x280, zero

#########################################
# State Enable Extension (Smstateen)
#########################################

# hstateen0
# name
# CHECK-INST: csrrs t1, hstateen0, zero
# CHECK-ENC: encoding: [0x73,0x23,0xc0,0x60]
# CHECK-INST-ALIAS: csrr t1, hstateen0
# uimm12
# CHECK-INST: csrrs t2, hstateen0, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xc0,0x60]
# CHECK-INST-ALIAS: csrr t2, hstateen0
# name
csrrs t1, hstateen0, zero
# uimm12
csrrs t2, 0x60C, zero

# hstateen1
# name
# CHECK-INST: csrrs t1, hstateen1, zero
# CHECK-ENC: encoding: [0x73,0x23,0xd0,0x60]
# CHECK-INST-ALIAS: csrr t1, hstateen1
# uimm12
# CHECK-INST: csrrs t2, hstateen1, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xd0,0x60]
# CHECK-INST-ALIAS: csrr t2, hstateen1
# name
csrrs t1, hstateen1, zero
# uimm12
csrrs t2, 0x60D, zero

# hstateen2
# name
# CHECK-INST: csrrs t1, hstateen2, zero
# CHECK-ENC: encoding: [0x73,0x23,0xe0,0x60]
# CHECK-INST-ALIAS: csrr t1, hstateen2
# uimm12
# CHECK-INST: csrrs t2, hstateen2, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xe0,0x60]
# CHECK-INST-ALIAS: csrr t2, hstateen2
# name
csrrs t1, hstateen2, zero
# uimm12
csrrs t2, 0x60E, zero

# hstateen3
# name
# CHECK-INST: csrrs t1, hstateen3, zero
# CHECK-ENC: encoding: [0x73,0x23,0xf0,0x60]
# CHECK-INST-ALIAS: csrr t1, hstateen3
# uimm12
# CHECK-INST: csrrs t2, hstateen3, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xf0,0x60]
# CHECK-INST-ALIAS: csrr t2, hstateen3
# name
csrrs t1, hstateen3, zero
# uimm12
csrrs t2, 0x60F, zero

#########################################
# Advanced Interrupt Architecture (Smaia and Ssaia)
#########################################

# hvien
# name
# CHECK-INST: csrrs t1, hvien, zero
# CHECK-ENC: encoding: [0x73,0x23,0x80,0x60]
# CHECK-INST-ALIAS: csrr t1, hvien
# uimm12
# CHECK-INST: csrrs t2, hvien, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x80,0x60]
# CHECK-INST-ALIAS: csrr t2, hvien
# name
csrrs t1, hvien, zero
# uimm12
csrrs t2, 0x608, zero

# hvictl
# name
# CHECK-INST: csrrs t1, hvictl, zero
# CHECK-ENC: encoding: [0x73,0x23,0x90,0x60]
# CHECK-INST-ALIAS: csrr t1, hvictl
# uimm12
# CHECK-INST: csrrs t2, hvictl, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x90,0x60]
# CHECK-INST-ALIAS: csrr t2, hvictl
# name
csrrs t1, hvictl, zero
# uimm12
csrrs t2, 0x609, zero

# hviprio1
# name
# CHECK-INST: csrrs t1, hviprio1, zero
# CHECK-ENC: encoding: [0x73,0x23,0x60,0x64]
# CHECK-INST-ALIAS: csrr t1, hviprio1
# uimm12
# CHECK-INST: csrrs t2, hviprio1, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x60,0x64]
# CHECK-INST-ALIAS: csrr t2, hviprio1
# name
csrrs t1, hviprio1, zero
# uimm12
csrrs t2, 0x646, zero

# hviprio2
# name
# CHECK-INST: csrrs t1, hviprio2, zero
# CHECK-ENC: encoding: [0x73,0x23,0x70,0x64]
# CHECK-INST-ALIAS: csrr t1, hviprio2
# uimm12
# CHECK-INST: csrrs t2, hviprio2, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x70,0x64]
# CHECK-INST-ALIAS: csrr t2, hviprio2
# name
csrrs t1, hviprio2, zero
# uimm12
csrrs t2, 0x647, zero

# vsiselect
# name
# CHECK-INST: csrrs t1, vsiselect, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0x25]
# CHECK-INST-ALIAS: csrr t1, vsiselect
# uimm12
# CHECK-INST: csrrs t2, vsiselect, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0x25]
# CHECK-INST-ALIAS: csrr t2, vsiselect
# name
csrrs t1, vsiselect, zero
# uimm12
csrrs t2, 0x250, zero

# vsireg
# name
# CHECK-INST: csrrs t1, vsireg, zero
# CHECK-ENC: encoding: [0x73,0x23,0x10,0x25]
# CHECK-INST-ALIAS: csrr t1, vsireg
# uimm12
# CHECK-INST: csrrs t2, vsireg, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x10,0x25]
# CHECK-INST-ALIAS: csrr t2, vsireg
# name
csrrs t1, vsireg, zero
# uimm12
csrrs t2, 0x251, zero

# vsireg2
# name
# CHECK-INST: csrrs t1, vsireg2, zero
# CHECK-ENC: encoding: [0x73,0x23,0x20,0x25]
# CHECK-INST-ALIAS: csrr t1, vsireg2
# uimm12
# CHECK-INST: csrrs t2, vsireg2, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x20,0x25]
# CHECK-INST-ALIAS: csrr t2, vsireg2
# name
csrrs t1, vsireg2, zero
# uimm12
csrrs t2, 0x252, zero

# vsireg3
# name
# CHECK-INST: csrrs t1, vsireg3, zero
# CHECK-ENC: encoding: [0x73,0x23,0x30,0x25]
# CHECK-INST-ALIAS: csrr t1, vsireg3
# uimm12
# CHECK-INST: csrrs t2, vsireg3, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x30,0x25]
# CHECK-INST-ALIAS: csrr t2, vsireg3
# name
csrrs t1, vsireg3, zero
# uimm12
csrrs t2, 0x253, zero

# vsireg4
# name
# CHECK-INST: csrrs t1, vsireg4, zero
# CHECK-ENC: encoding: [0x73,0x23,0x50,0x25]
# CHECK-INST-ALIAS: csrr t1, vsireg4
# uimm12
# CHECK-INST: csrrs t2, vsireg4, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x50,0x25]
# CHECK-INST-ALIAS: csrr t2, vsireg4
# name
csrrs t1, vsireg4, zero
# uimm12
csrrs t2, 0x255, zero

# vsireg5
# name
# CHECK-INST: csrrs t1, vsireg5, zero
# CHECK-ENC: encoding: [0x73,0x23,0x60,0x25]
# CHECK-INST-ALIAS: csrr t1, vsireg5
# uimm12
# CHECK-INST: csrrs t2, vsireg5, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x60,0x25]
# CHECK-INST-ALIAS: csrr t2, vsireg5
# name
csrrs t1, vsireg5, zero
# uimm12
csrrs t2, 0x256, zero

# vsireg6
# name
# CHECK-INST: csrrs t1, vsireg6, zero
# CHECK-ENC: encoding: [0x73,0x23,0x70,0x25]
# CHECK-INST-ALIAS: csrr t1, vsireg6
# uimm12
# CHECK-INST: csrrs t2, vsireg6, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x70,0x25]
# CHECK-INST-ALIAS: csrr t2, vsireg6
# name
csrrs t1, vsireg6, zero
# uimm12
csrrs t2, 0x257, zero

# vstopei
# name
# CHECK-INST: csrrs t1, vstopei, zero
# CHECK-ENC: encoding: [0x73,0x23,0xc0,0x25]
# CHECK-INST-ALIAS: csrr t1, vstopei
# uimm12
# CHECK-INST: csrrs t2, vstopei, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xc0,0x25]
# CHECK-INST-ALIAS: csrr t2, vstopei
# name
csrrs t1, vstopei, zero
# uimm12
csrrs t2, 0x25C, zero

# vstopi
# name
# CHECK-INST: csrrs t1, vstopi, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0xeb]
# CHECK-INST-ALIAS: csrr t1, vstopi
# uimm12
# CHECK-INST: csrrs t2, vstopi, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0xeb]
# CHECK-INST-ALIAS: csrr t2, vstopi
# name
csrrs t1, vstopi, zero
# uimm12
csrrs t2, 0xEB0, zero
