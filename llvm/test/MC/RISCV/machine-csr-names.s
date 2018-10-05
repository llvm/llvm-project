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
# Machine Information Registers
##################################

# mvendorid
# name
# CHECK-INST: csrrs t1, mvendorid, zero
# CHECK-ENC: encoding: [0x73,0x23,0x10,0xf1]
# CHECK-INST-ALIAS: csrr t1, mvendorid
# uimm12
# CHECK-INST: csrrs t2, mvendorid, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x10,0xf1]
# CHECK-INST-ALIAS: csrr t2, mvendorid
# name
csrrs t1, mvendorid, zero
# uimm12
csrrs t2, 0xF11, zero

# marchid
# name
# CHECK-INST: csrrs t1, marchid, zero
# CHECK-ENC: encoding: [0x73,0x23,0x20,0xf1]
# CHECK-INST-ALIAS: csrr t1, marchid
# uimm12
# CHECK-INST: csrrs t2, marchid, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x20,0xf1]
# CHECK-INST-ALIAS: csrr t2, marchid
# name
csrrs t1, marchid, zero
# uimm12
csrrs t2, 0xF12, zero

# mimpid
# name
# CHECK-INST: csrrs t1, mimpid, zero
# CHECK-ENC: encoding: [0x73,0x23,0x30,0xf1]
# CHECK-INST-ALIAS: csrr t1, mimpid
# uimm12
# CHECK-INST: csrrs t2, mimpid, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x30,0xf1]
# CHECK-INST-ALIAS: csrr t2, mimpid
csrrs t1, mimpid, zero
# uimm12
csrrs t2, 0xF13, zero

# mhartid
# name
# CHECK-INST: csrrs t1, mhartid, zero
# CHECK-ENC: encoding: [0x73,0x23,0x40,0xf1]
# CHECK-INST-ALIAS: csrr t1, mhartid
# uimm12
# CHECK-INST: csrrs t2, mhartid, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x40,0xf1]
# CHECK-INST-ALIAS: csrr t2, mhartid
# name
csrrs t1, mhartid, zero
# uimm12
csrrs t2, 0xF14, zero

##################################
# Machine Trap Setup
##################################

# mstatus
# name
# CHECK-INST: csrrs t1, mstatus, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0x30]
# CHECK-INST-ALIAS: csrr t1, mstatus
# uimm12
# CHECK-INST: csrrs t2, mstatus, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0x30]
# CHECK-INST-ALIAS: csrr t2, mstatus
# name
csrrs t1, mstatus, zero
# uimm12
csrrs t2, 0x300, zero

# misa
# name
# CHECK-INST: csrrs t1, misa, zero
# CHECK-ENC: encoding: [0x73,0x23,0x10,0x30]
# CHECK-INST-ALIAS: csrr t1, misa
# uimm12
# CHECK-INST: csrrs t2, misa, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x10,0x30]
# CHECK-INST-ALIAS: csrr t2, misa
# name
csrrs t1, misa, zero
# uimm12
csrrs t2, 0x301, zero

# medeleg
# name
# CHECK-INST: csrrs t1, medeleg, zero
# CHECK-ENC: encoding: [0x73,0x23,0x20,0x30]
# CHECK-INST-ALIAS: csrr t1, medeleg
# uimm12
# CHECK-INST: csrrs t2, medeleg, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x20,0x30]
# CHECK-INST-ALIAS: csrr t2, medeleg
# name
csrrs t1, medeleg, zero
# uimm12
csrrs t2, 0x302, zero
# aliases

# mideleg
# name
# CHECK-INST: csrrs t1, mideleg, zero
# CHECK-ENC: encoding: [0x73,0x23,0x30,0x30]
# CHECK-INST-ALIAS: csrr t1, mideleg
# uimm12
# CHECK-INST: csrrs t2, mideleg, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x30,0x30]
# CHECK-INST-ALIAS: csrr t2, mideleg
# aliases
# name
csrrs t1, mideleg, zero
# uimm12
csrrs t2, 0x303, zero

# mie
# name
# CHECK-INST: csrrs t1, mie, zero
# CHECK-ENC: encoding: [0x73,0x23,0x40,0x30]
# CHECK-INST-ALIAS: csrr t1, mie
# uimm12
# CHECK-INST: csrrs t2, mie, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x40,0x30]
# CHECK-INST-ALIAS: csrr t2, mie
# name
csrrs t1, mie, zero
# uimm12
csrrs t2, 0x304, zero

# mtvec
# name
# CHECK-INST: csrrs t1, mtvec, zero
# CHECK-ENC: encoding: [0x73,0x23,0x50,0x30]
# CHECK-INST-ALIAS: csrr t1, mtvec
# uimm12
# CHECK-INST: csrrs t2, mtvec, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x50,0x30]
# CHECK-INST-ALIAS: csrr t2, mtvec
# name
csrrs t1, mtvec, zero
# uimm12
csrrs t2, 0x305, zero

# mcounteren
# name
# CHECK-INST: csrrs t1, mcounteren, zero
# CHECK-ENC: encoding: [0x73,0x23,0x60,0x30]
# CHECK-INST-ALIAS: csrr t1, mcounteren
# uimm12
# CHECK-INST: csrrs t2, mcounteren, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x60,0x30]
# CHECK-INST-ALIAS: csrr t2, mcounteren
# name
csrrs t1, mcounteren, zero
# uimm12
csrrs t2, 0x306, zero

# mscratch
# name
# CHECK-INST: csrrs t1, mscratch, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0x34]
# CHECK-INST-ALIAS: csrr t1, mscratch
# uimm12
# CHECK-INST: csrrs t2, mscratch, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0x34]
# CHECK-INST-ALIAS: csrr t2, mscratch
# name
csrrs t1, mscratch, zero
# uimm12
csrrs t2, 0x340, zero

# mepc
# name
# CHECK-INST: csrrs t1, mepc, zero
# CHECK-ENC: encoding: [0x73,0x23,0x10,0x34]
# CHECK-INST-ALIAS: csrr t1, mepc
# uimm12
# CHECK-INST: csrrs t2, mepc, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x10,0x34]
# CHECK-INST-ALIAS: csrr t2, mepc
# name
csrrs t1, mepc, zero
# uimm12
csrrs t2, 0x341, zero

# mcause
# name
# CHECK-INST: csrrs t1, mcause, zero
# CHECK-ENC: encoding: [0x73,0x23,0x20,0x34]
# CHECK-INST-ALIAS: csrr t1, mcause
# uimm12
# CHECK-INST: csrrs t2, mcause, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x20,0x34]
# CHECK-INST-ALIAS: csrr t2, mcause
# name
csrrs t1, mcause, zero
# uimm12
csrrs t2, 0x342, zero

# mtval
# name
# CHECK-INST: csrrs t1, mtval, zero
# CHECK-ENC: encoding: [0x73,0x23,0x30,0x34]
# CHECK-INST-ALIAS: csrr t1, mtval
# uimm12
# CHECK-INST: csrrs t2, mtval, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x30,0x34]
# CHECK-INST-ALIAS: csrr t2, mtval
# name
csrrs t1, mtval, zero
# uimm12
csrrs t2, 0x343, zero

# mip
# name
# CHECK-INST: csrrs t1, mip, zero
# CHECK-ENC: encoding: [0x73,0x23,0x40,0x34]
# CHECK-INST-ALIAS: csrr t1, mip
# uimm12
# CHECK-INST: csrrs t2, mip, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x40,0x34]
# CHECK-INST-ALIAS: csrr t2, mip
# name
csrrs t1, mip, zero
# uimm12
csrrs t2, 0x344, zero

######################################
# Machine Protection and Translation
######################################
# Tests for pmpcfg1, pmpcfg2 in rv32-machine-csr-names.s

# pmpcfg0
# name
# CHECK-INST: csrrs t1, pmpcfg0, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0x3a]
# CHECK-INST-ALIAS: csrr t1, pmpcfg0
# uimm12
# CHECK-INST: csrrs t2, pmpcfg0, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0x3a]
# CHECK-INST-ALIAS: csrr t2, pmpcfg0
# name
csrrs t1, pmpcfg0, zero
# uimm12
csrrs t2, 0x3A0, zero

# pmpcfg2
# name
# CHECK-INST: csrrs t1, pmpcfg2, zero
# CHECK-ENC: encoding: [0x73,0x23,0x20,0x3a]
# CHECK-INST-ALIAS: csrr t1, pmpcfg2
# uimm12
# CHECK-INST: csrrs t2, pmpcfg2, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x20,0x3a]
# CHECK-INST-ALIAS: csrr t2, pmpcfg2
# name
csrrs t1, pmpcfg2, zero
# uimm12
csrrs t2, 0x3A2, zero


######################################
# Machine Counter and Timers
######################################
# mcycle
# name
# CHECK-INST: csrrs t1, mcycle, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0xb0]
# CHECK-INST-ALIAS: csrr t1, mcycle
# uimm12
# CHECK-INST: csrrs t2, mcycle, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0xb0]
# CHECK-INST-ALIAS: csrr t2, mcycle
csrrs t1, mcycle, zero
# uimm12
csrrs t2, 0xB00, zero

# minstret
# name
# CHECK-INST: csrrs t1, minstret, zero
# CHECK-ENC: encoding: [0x73,0x23,0x20,0xb0]
# CHECK-INST-ALIAS: csrr t1, minstret
# uimm12
# CHECK-INST: csrrs t2, minstret, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x20,0xb0]
# CHECK-INST-ALIAS: csrr t2, minstret
# name
csrrs t1, minstret, zero
# uimm12
csrrs t2, 0xB02, zero


######################################################
# Debug and Trace Registers (shared with Debug Mode)
######################################################
# tselect
# name
# CHECK-INST: csrrs t1, tselect, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0x7a]
# CHECK-INST-ALIAS: csrr t1, tselect
# uimm12
# CHECK-INST: csrrs t2, tselect, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0x7a]
# CHECK-INST-ALIAS: csrr t2, tselect
# name
csrrs t1, tselect, zero
# uimm12
csrrs t2, 0x7A0, zero

# tdata1
# name
# CHECK-INST: csrrs t1, tdata1, zero
# CHECK-ENC: encoding: [0x73,0x23,0x10,0x7a]
# CHECK-INST-ALIAS: csrr t1, tdata1
# uimm12
# CHECK-INST: csrrs t2, tdata1, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x10,0x7a]
# CHECK-INST-ALIAS: csrr t2, tdata1
# name
csrrs t1, tdata1, zero
# uimm12
csrrs t2, 0x7A1, zero

# tdata2
# name
# CHECK-INST: csrrs t1, tdata2, zero
# CHECK-ENC: encoding: [0x73,0x23,0x20,0x7a]
# CHECK-INST-ALIAS: csrr t1, tdata2
# uimm12
# CHECK-INST: csrrs t2, tdata2, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x20,0x7a]
# CHECK-INST-ALIAS: csrr t2, tdata2
csrrs t1, tdata2, zero
# uimm12
csrrs t2, 0x7A2, zero

#tdata3
# name
# CHECK-INST: csrrs t1, tdata3, zero
# CHECK-ENC: encoding: [0x73,0x23,0x30,0x7a]
# CHECK-INST-ALIAS: csrr t1, tdata3
# uimm12
# CHECK-INST: csrrs t2, tdata3, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x30,0x7a]
# CHECK-INST-ALIAS: csrr t2, tdata3
# name
csrrs t1, tdata3, zero
# uimm12
csrrs t2, 0x7A3, zero

#######################
# Debug Mode Registers
########################
# dcsr
# name
# CHECK-INST: csrrs t1, dcsr, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0x7b]
# CHECK-INST-ALIAS: csrr t1, dcsr
# uimm12
# CHECK-INST: csrrs t2, dcsr, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0x7b]
# CHECK-INST-ALIAS: csrr t2, dcsr
# name
csrrs t1, dcsr, zero
# uimm12
csrrs t2, 0x7B0, zero

# dpc
# name
# CHECK-INST: csrrs t1, dpc, zero
# CHECK-ENC: encoding: [0x73,0x23,0x10,0x7b]
# CHECK-INST-ALIAS: csrr t1, dpc
# uimm12
# CHECK-INST: csrrs t2, dpc, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x10,0x7b]
# CHECK-INST-ALIAS: csrr t2, dpc
# name
csrrs t1, dpc, zero
# uimm12
csrrs t2, 0x7B1, zero

# dscratch
# name
# CHECK-INST: csrrs t1, dscratch, zero
# CHECK-ENC: encoding: [0x73,0x23,0x20,0x7b]
# CHECK-INST-ALIAS: csrr t1, dscratch
# uimm12
# CHECK-INST: csrrs t2, dscratch, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x20,0x7b]
# CHECK-INST-ALIAS: csrr t2, dscratch
# name
csrrs t1, dscratch, zero
# uimm12
csrrs t2, 0x7B2, zero

# mhpmcounter3
# name
# CHECK-INST: csrrs t1, mhpmcounter3, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x30,0xb0]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter3
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter3, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x30,0xb0]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter3
# name
csrrs t1, mhpmcounter3, zero
# uimm12
csrrs t2, 0xB03, zero

# mhpmcounter4
# name
# CHECK-INST: csrrs t1, mhpmcounter4, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x40,0xb0]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter4
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter4, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x40,0xb0]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter4
# name
csrrs t1, mhpmcounter4, zero
# uimm12
csrrs t2, 0xB04, zero

# mhpmcounter5
# name
# CHECK-INST: csrrs t1, mhpmcounter5, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x50,0xb0]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter5
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter5, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x50,0xb0]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter5
# name
csrrs t1, mhpmcounter5, zero
# uimm12
csrrs t2, 0xB05, zero

# mhpmcounter6
# name
# CHECK-INST: csrrs t1, mhpmcounter6, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x60,0xb0]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter6
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter6, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x60,0xb0]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter6
# name
csrrs t1, mhpmcounter6, zero
# uimm12
csrrs t2, 0xB06, zero

# mhpmcounter7
# name
# CHECK-INST: csrrs t1, mhpmcounter7, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x70,0xb0]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter7
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter7, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x70,0xb0]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter7
# name
csrrs t1, mhpmcounter7, zero
# uimm12
csrrs t2, 0xB07, zero

# mhpmcounter8
# name
# CHECK-INST: csrrs t1, mhpmcounter8, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x80,0xb0]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter8
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter8, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x80,0xb0]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter8
# name
csrrs t1, mhpmcounter8, zero
# uimm12
csrrs t2, 0xB08, zero

# mhpmcounter9
# name
# CHECK-INST: csrrs t1, mhpmcounter9, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x90,0xb0]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter9
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter9, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x90,0xb0]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter9
# name
csrrs t1, mhpmcounter9, zero
# uimm12
csrrs t2, 0xB09, zero

# mhpmcounter10
# name
# CHECK-INST: csrrs t1, mhpmcounter10, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xa0,0xb0]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter10
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter10, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xa0,0xb0]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter10
# name
csrrs t1, mhpmcounter10, zero
# uimm12
csrrs t2, 0xB0A, zero

# mhpmcounter11
# name
# CHECK-INST: csrrs t1, mhpmcounter11, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xb0,0xb0]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter11
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter11, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xb0,0xb0]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter11
# name
csrrs t1, mhpmcounter11, zero
# uimm12
csrrs t2, 0xB0B, zero

# mhpmcounter12
# name
# CHECK-INST: csrrs t1, mhpmcounter12, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xc0,0xb0]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter12
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter12, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xc0,0xb0]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter12
# name
csrrs t1, mhpmcounter12, zero
# uimm12
csrrs t2, 0xB0C, zero

# mhpmcounter13
# name
# CHECK-INST: csrrs t1, mhpmcounter13, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xd0,0xb0]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter13
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter13, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xd0,0xb0]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter13
# name
csrrs t1, mhpmcounter13, zero
# uimm12
csrrs t2, 0xB0D, zero

# mhpmcounter14
# name
# CHECK-INST: csrrs t1, mhpmcounter14, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xe0,0xb0]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter14
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter14, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xe0,0xb0]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter14
# name
csrrs t1, mhpmcounter14, zero
# uimm12
csrrs t2, 0xB0E, zero

# mhpmcounter15
# name
# CHECK-INST: csrrs t1, mhpmcounter15, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xf0,0xb0]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter15
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter15, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xf0,0xb0]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter15
# name
csrrs t1, mhpmcounter15, zero
# uimm12
csrrs t2, 0xB0F, zero

# mhpmcounter16
# name
# CHECK-INST: csrrs t1, mhpmcounter16, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x00,0xb1]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter16
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter16, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x00,0xb1]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter16
# name
csrrs t1, mhpmcounter16, zero
# uimm12
csrrs t2, 0xB10, zero

# mhpmcounter17
# name
# CHECK-INST: csrrs t1, mhpmcounter17, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x10,0xb1]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter17
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter17, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x10,0xb1]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter17
# name
csrrs t1, mhpmcounter17, zero
# uimm12
csrrs t2, 0xB11, zero

# mhpmcounter18
# name
# CHECK-INST: csrrs t1, mhpmcounter18, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x20,0xb1]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter18
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter18, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x20,0xb1]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter18
# name
csrrs t1, mhpmcounter18, zero
# uimm12
csrrs t2, 0xB12, zero

# mhpmcounter19
# name
# CHECK-INST: csrrs t1, mhpmcounter19, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x30,0xb1]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter19
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter19, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x30,0xb1]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter19
# name
csrrs t1, mhpmcounter19, zero
# uimm12
csrrs t2, 0xB13, zero

# mhpmcounter20
# name
# CHECK-INST: csrrs t1, mhpmcounter20, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x40,0xb1]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter20
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter20, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x40,0xb1]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter20
# name
csrrs t1, mhpmcounter20, zero
# uimm12
csrrs t2, 0xB14, zero

# mhpmcounter21
# name
# CHECK-INST: csrrs t1, mhpmcounter21, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x50,0xb1]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter21
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter21, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x50,0xb1]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter21
# name
csrrs t1, mhpmcounter21, zero
# uimm12
csrrs t2, 0xB15, zero

# mhpmcounter22
# name
# CHECK-INST: csrrs t1, mhpmcounter22, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x60,0xb1]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter22
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter22, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x60,0xb1]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter22
# name
csrrs t1, mhpmcounter22, zero
# uimm12
csrrs t2, 0xB16, zero

# mhpmcounter23
# name
# CHECK-INST: csrrs t1, mhpmcounter23, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x70,0xb1]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter23
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter23, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x70,0xb1]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter23
# name
csrrs t1, mhpmcounter23, zero
# uimm12
csrrs t2, 0xB17, zero

# mhpmcounter24
# name
# CHECK-INST: csrrs t1, mhpmcounter24, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x80,0xb1]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter24
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter24, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x80,0xb1]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter24
# name
csrrs t1, mhpmcounter24, zero
# uimm12
csrrs t2, 0xB18, zero

# mhpmcounter25
# name
# CHECK-INST: csrrs t1, mhpmcounter25, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x90,0xb1]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter25
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter25, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x90,0xb1]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter25
# name
csrrs t1, mhpmcounter25, zero
# uimm12
csrrs t2, 0xB19, zero

# mhpmcounter26
# name
# CHECK-INST: csrrs t1, mhpmcounter26, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xa0,0xb1]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter26
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter26, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xa0,0xb1]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter26
# name
csrrs t1, mhpmcounter26, zero
# uimm12
csrrs t2, 0xB1A, zero

# mhpmcounter27
# name
# CHECK-INST: csrrs t1, mhpmcounter27, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xb0,0xb1]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter27
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter27, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xb0,0xb1]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter27
# name
csrrs t1, mhpmcounter27, zero
# uimm12
csrrs t2, 0xB1B, zero

# mhpmcounter28
# name
# CHECK-INST: csrrs t1, mhpmcounter28, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xc0,0xb1]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter28
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter28, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xc0,0xb1]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter28
# name
csrrs t1, mhpmcounter28, zero
# uimm12
csrrs t2, 0xB1C, zero

# mhpmcounter29
# name
# CHECK-INST: csrrs t1, mhpmcounter29, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xd0,0xb1]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter29
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter29, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xd0,0xb1]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter29
# name
csrrs t1, mhpmcounter29, zero
# uimm12
csrrs t2, 0xB1D, zero

# mhpmcounter30
# name
# CHECK-INST: csrrs t1, mhpmcounter30, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xe0,0xb1]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter30
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter30, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xe0,0xb1]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter30
# name
csrrs t1, mhpmcounter30, zero
# uimm12
csrrs t2, 0xB1E, zero

# mhpmcounter31
# name
# CHECK-INST: csrrs t1, mhpmcounter31, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xf0,0xb1]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter31
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter31, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xf0,0xb1]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter31
# name
csrrs t1, mhpmcounter31, zero
# uimm12
csrrs t2, 0xB1F, zero


######################################
# Machine Counter Setup
######################################
# mhpmevent3
# name
# CHECK-INST: csrrs t1, mhpmevent3, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x30,0x32]
# CHECK-INST-ALIAS: csrr t1, mhpmevent3
# uimm12
# CHECK-INST: csrrs t2, mhpmevent3, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x30,0x32]
# CHECK-INST-ALIAS: csrr t2, mhpmevent3
# name
csrrs t1, mhpmevent3, zero
# uimm12
csrrs t2, 0x323, zero

# mhpmevent4
# name
# CHECK-INST: csrrs t1, mhpmevent4, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x40,0x32]
# CHECK-INST-ALIAS: csrr t1, mhpmevent4
# uimm12
# CHECK-INST: csrrs t2, mhpmevent4, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x40,0x32]
# CHECK-INST-ALIAS: csrr t2, mhpmevent4
# name
csrrs t1, mhpmevent4, zero
# uimm12
csrrs t2, 0x324, zero

# mhpmevent5
# name
# CHECK-INST: csrrs t1, mhpmevent5, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x50,0x32]
# CHECK-INST-ALIAS: csrr t1, mhpmevent5
# uimm12
# CHECK-INST: csrrs t2, mhpmevent5, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x50,0x32]
# CHECK-INST-ALIAS: csrr t2, mhpmevent5
# name
csrrs t1, mhpmevent5, zero
# uimm12
csrrs t2, 0x325, zero

# mhpmevent6
# name
# CHECK-INST: csrrs t1, mhpmevent6, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x60,0x32]
# CHECK-INST-ALIAS: csrr t1, mhpmevent6
# uimm12
# CHECK-INST: csrrs t2, mhpmevent6, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x60,0x32]
# CHECK-INST-ALIAS: csrr t2, mhpmevent6
# name
csrrs t1, mhpmevent6, zero
# uimm12
csrrs t2, 0x326, zero

# mhpmevent7
# name
# CHECK-INST: csrrs t1, mhpmevent7, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x70,0x32]
# CHECK-INST-ALIAS: csrr t1, mhpmevent7
# uimm12
# CHECK-INST: csrrs t2, mhpmevent7, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x70,0x32]
# CHECK-INST-ALIAS: csrr t2, mhpmevent7
# name
csrrs t1, mhpmevent7, zero
# uimm12
csrrs t2, 0x327, zero

# mhpmevent8
# name
# CHECK-INST: csrrs t1, mhpmevent8, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x80,0x32]
# CHECK-INST-ALIAS: csrr t1, mhpmevent8
# uimm12
# CHECK-INST: csrrs t2, mhpmevent8, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x80,0x32]
# CHECK-INST-ALIAS: csrr t2, mhpmevent8
# name
csrrs t1, mhpmevent8, zero
# uimm12
csrrs t2, 0x328, zero

# mhpmevent9
# name
# CHECK-INST: csrrs t1, mhpmevent9, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x90,0x32]
# CHECK-INST-ALIAS: csrr t1, mhpmevent9
# uimm12
# CHECK-INST: csrrs t2, mhpmevent9, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x90,0x32]
# CHECK-INST-ALIAS: csrr t2, mhpmevent9
# name
csrrs t1, mhpmevent9, zero
# uimm12
csrrs t2, 0x329, zero

# mhpmevent10
# name
# CHECK-INST: csrrs t1, mhpmevent10, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xa0,0x32]
# CHECK-INST-ALIAS: csrr t1, mhpmevent10
# uimm12
# CHECK-INST: csrrs t2, mhpmevent10, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xa0,0x32]
# CHECK-INST-ALIAS: csrr t2, mhpmevent10
# name
csrrs t1, mhpmevent10, zero
# uimm12
csrrs t2, 0x32A, zero

# mhpmevent11
# name
# CHECK-INST: csrrs t1, mhpmevent11, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xb0,0x32]
# CHECK-INST-ALIAS: csrr t1, mhpmevent11
# uimm12
# CHECK-INST: csrrs t2, mhpmevent11, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xb0,0x32]
# CHECK-INST-ALIAS: csrr t2, mhpmevent11
# name
csrrs t1, mhpmevent11, zero
# uimm12
csrrs t2, 0x32B, zero

# mhpmevent12
# name
# CHECK-INST: csrrs t1, mhpmevent12, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xc0,0x32]
# CHECK-INST-ALIAS: csrr t1, mhpmevent12
# uimm12
# CHECK-INST: csrrs t2, mhpmevent12, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xc0,0x32]
# CHECK-INST-ALIAS: csrr t2, mhpmevent12
# name
csrrs t1, mhpmevent12, zero
# uimm12
csrrs t2, 0x32C, zero

# mhpmevent13
# name
# CHECK-INST: csrrs t1, mhpmevent13, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xd0,0x32]
# CHECK-INST-ALIAS: csrr t1, mhpmevent13
# uimm12
# CHECK-INST: csrrs t2, mhpmevent13, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xd0,0x32]
# CHECK-INST-ALIAS: csrr t2, mhpmevent13
# name
csrrs t1, mhpmevent13, zero
# uimm12
csrrs t2, 0x32D, zero

# mhpmevent14
# name
# CHECK-INST: csrrs t1, mhpmevent14, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xe0,0x32]
# CHECK-INST-ALIAS: csrr t1, mhpmevent14
# uimm12

# CHECK-INST: csrrs t2, mhpmevent14, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xe0,0x32]
# CHECK-INST-ALIAS: csrr t2, mhpmevent14
# name
csrrs t1, mhpmevent14, zero
# uimm12
csrrs t2, 0x32E, zero

# mhpmevent15
# name
# CHECK-INST: csrrs t1, mhpmevent15, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xf0,0x32]
# CHECK-INST-ALIAS: csrr t1, mhpmevent15
# uimm12
# CHECK-INST: csrrs t2, mhpmevent15, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xf0,0x32]
# CHECK-INST-ALIAS: csrr t2, mhpmevent15
# name
csrrs t1, mhpmevent15, zero
# uimm12
csrrs t2, 0x32F, zero

# mhpmevent16
# name
# CHECK-INST: csrrs t1, mhpmevent16, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x00,0x33]
# CHECK-INST-ALIAS: csrr t1, mhpmevent16
# uimm12
# CHECK-INST: csrrs t2, mhpmevent16, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x00,0x33]
# CHECK-INST-ALIAS: csrr t2, mhpmevent16
# name
csrrs t1, mhpmevent16, zero
# uimm12
csrrs t2, 0x330, zero

# mhpmevent17
# name
# CHECK-INST: csrrs t1, mhpmevent17, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x10,0x33]
# CHECK-INST-ALIAS: csrr t1, mhpmevent17
# uimm12
# CHECK-INST: csrrs t2, mhpmevent17, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x10,0x33]
# CHECK-INST-ALIAS: csrr t2, mhpmevent17
# name
csrrs t1, mhpmevent17, zero
# uimm12
csrrs t2, 0x331, zero

# mhpmevent18
# name
# CHECK-INST: csrrs t1, mhpmevent18, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x20,0x33]
# CHECK-INST-ALIAS: csrr t1, mhpmevent18
# uimm12
# CHECK-INST: csrrs t2, mhpmevent18, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x20,0x33]
# CHECK-INST-ALIAS: csrr t2, mhpmevent18
# name
csrrs t1, mhpmevent18, zero
# uimm12
csrrs t2, 0x332, zero

# mhpmevent19
# name
# CHECK-INST: csrrs t1, mhpmevent19, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x30,0x33]
# CHECK-INST-ALIAS: csrr t1, mhpmevent19
# uimm12
# CHECK-INST: csrrs t2, mhpmevent19, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x30,0x33]
# CHECK-INST-ALIAS: csrr t2, mhpmevent19
# name
csrrs t1, mhpmevent19, zero
# uimm12
csrrs t2, 0x333, zero

# mhpmevent20
# name
# CHECK-INST: csrrs t1, mhpmevent20, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x40,0x33]
# CHECK-INST-ALIAS: csrr t1, mhpmevent20
# uimm12
# CHECK-INST: csrrs t2, mhpmevent20, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x40,0x33]
# CHECK-INST-ALIAS: csrr t2, mhpmevent20
# name
csrrs t1, mhpmevent20, zero
# uimm12
csrrs t2, 0x334, zero

# mhpmevent21
# name
# CHECK-INST: csrrs t1, mhpmevent21, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x50,0x33]
# CHECK-INST-ALIAS: csrr t1, mhpmevent21
# uimm12
# CHECK-INST: csrrs t2, mhpmevent21, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x50,0x33]
# CHECK-INST-ALIAS: csrr t2, mhpmevent21
# name
csrrs t1, mhpmevent21, zero
# uimm12
csrrs t2, 0x335, zero

# mhpmevent22
# name
# CHECK-INST: csrrs t1, mhpmevent22, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x60,0x33]
# CHECK-INST-ALIAS: csrr t1, mhpmevent22
# uimm12
# CHECK-INST: csrrs t2, mhpmevent22, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x60,0x33]
# CHECK-INST-ALIAS: csrr t2, mhpmevent22
# name
csrrs t1, mhpmevent22, zero
# uimm12
csrrs t2, 0x336, zero

# mhpmevent23
# name
# CHECK-INST: csrrs t1, mhpmevent23, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x70,0x33]
# CHECK-INST-ALIAS: csrr t1, mhpmevent23
# uimm12
# CHECK-INST: csrrs t2, mhpmevent23, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x70,0x33]
# CHECK-INST-ALIAS: csrr t2, mhpmevent23
# name
csrrs t1, mhpmevent23, zero
# uimm12
csrrs t2, 0x337, zero

# mhpmevent24
# name
# CHECK-INST: csrrs t1, mhpmevent24, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x80,0x33]
# CHECK-INST-ALIAS: csrr t1, mhpmevent24
# uimm12
# CHECK-INST: csrrs t2, mhpmevent24, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x80,0x33]
# CHECK-INST-ALIAS: csrr t2, mhpmevent24
# name
csrrs t1, mhpmevent24, zero
# uimm12
csrrs t2, 0x338, zero

# mhpmevent25
# name
# CHECK-INST: csrrs t1, mhpmevent25, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x90,0x33]
# CHECK-INST-ALIAS: csrr t1, mhpmevent25
# uimm12
# CHECK-INST: csrrs t2, mhpmevent25, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x90,0x33]
# CHECK-INST-ALIAS: csrr t2, mhpmevent25
# name
csrrs t1, mhpmevent25, zero
# uimm12
csrrs t2, 0x339, zero

# mhpmevent26
# name
# CHECK-INST: csrrs t1, mhpmevent26, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xa0,0x33]
# CHECK-INST-ALIAS: csrr t1, mhpmevent26
# uimm12
# CHECK-INST: csrrs t2, mhpmevent26, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xa0,0x33]
# CHECK-INST-ALIAS: csrr t2, mhpmevent26
# name
csrrs t1, mhpmevent26, zero
# uimm12
csrrs t2, 0x33A, zero

# mhpmevent27
# name
# CHECK-INST: csrrs t1, mhpmevent27, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xb0,0x33]
# CHECK-INST-ALIAS: csrr t1, mhpmevent27
# uimm12
# CHECK-INST: csrrs t2, mhpmevent27, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xb0,0x33]
# CHECK-INST-ALIAS: csrr t2, mhpmevent27
# name
csrrs t1, mhpmevent27, zero
# uimm12
csrrs t2, 0x33B, zero

# mhpmevent28
# name
# CHECK-INST: csrrs t1, mhpmevent28, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xc0,0x33]
# CHECK-INST-ALIAS: csrr t1, mhpmevent28
# uimm12
# CHECK-INST: csrrs t2, mhpmevent28, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xc0,0x33]
# CHECK-INST-ALIAS: csrr t2, mhpmevent28
# name
csrrs t1, mhpmevent28, zero
# uimm12
csrrs t2, 0x33C, zero

# mhpmevent29
# name
# CHECK-INST: csrrs t1, mhpmevent29, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xd0,0x33]
# CHECK-INST-ALIAS: csrr t1, mhpmevent29
# uimm12
# CHECK-INST: csrrs t2, mhpmevent29, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xd0,0x33]
# CHECK-INST-ALIAS: csrr t2, mhpmevent29
# name
csrrs t1, mhpmevent29, zero
# uimm12
csrrs t2, 0x33D, zero

# mhpmevent30
# name
# CHECK-INST: csrrs t1, mhpmevent30, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xe0,0x33]
# CHECK-INST-ALIAS: csrr t1, mhpmevent30
# uimm12
# CHECK-INST: csrrs t2, mhpmevent30, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xe0,0x33]
# CHECK-INST-ALIAS: csrr t2, mhpmevent30
# name
csrrs t1, mhpmevent30, zero
# uimm12
csrrs t2, 0x33E, zero

# mhpmevent31
# name
# CHECK-INST: csrrs t1, mhpmevent31, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xf0,0x33]
# CHECK-INST-ALIAS: csrr t1, mhpmevent31
# uimm12
# CHECK-INST: csrrs t2, mhpmevent31, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xf0,0x33]
# CHECK-INST-ALIAS: csrr t2, mhpmevent31
# name
csrrs t1, mhpmevent31, zero
# uimm12
csrrs t2, 0x33F, zero
