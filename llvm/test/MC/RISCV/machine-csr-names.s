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

# mconfigptr
# name
# CHECK-INST: csrrs t1, mconfigptr, zero
# CHECK-ENC: encoding: [0x73,0x23,0x50,0xf1]
# CHECK-INST-ALIAS: csrr t1, mconfigptr
# uimm12
# CHECK-INST: csrrs t2, mconfigptr, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x50,0xf1]
# CHECK-INST-ALIAS: csrr t2, mconfigptr
# name
csrrs t1, mconfigptr, zero
# uimm12
csrrs t2, 0xF15, zero

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

# mtinst
# name
# CHECK-INST: csrrs t1, mtinst, zero
# CHECK-ENC: encoding: [0x73,0x23,0xa0,0x34]
# CHECK-INST-ALIAS: csrr t1, mtinst
# uimm12
# CHECK-INST: csrrs t2, mtinst, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xa0,0x34]
# CHECK-INST-ALIAS: csrr t2, mtinst
# name
csrrs t1, mtinst, zero
# uimm12
csrrs t2, 0x34A, zero

# mtval2
# name
# CHECK-INST: csrrs t1, mtval2, zero
# CHECK-ENC: encoding: [0x73,0x23,0xb0,0x34]
# CHECK-INST-ALIAS: csrr t1, mtval2
# uimm12
# CHECK-INST: csrrs t2, mtval2, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xb0,0x34]
# CHECK-INST-ALIAS: csrr t2, mtval2
# name
csrrs t1, mtval2, zero
# uimm12
csrrs t2, 0x34B, zero

#########################
# Machine Configuration
#########################

# menvcfg
# name
# CHECK-INST: csrrs t1, menvcfg, zero
# CHECK-ENC: encoding: [0x73,0x23,0xa0,0x30]
# CHECK-INST-ALIAS: csrr t1, menvcfg
# uimm12
# CHECK-INST: csrrs t2, menvcfg, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xa0,0x30]
# CHECK-INST-ALIAS: csrr t2, menvcfg
# name
csrrs t1, menvcfg, zero
# uimm12
csrrs t2, 0x30A, zero

# mseccfg
# name
# CHECK-INST: csrrs t1, mseccfg, zero
# CHECK-ENC: encoding: [0x73,0x23,0x70,0x74]
# CHECK-INST-ALIAS: csrr t1, mseccfg
# uimm12
# CHECK-INST: csrrs t2, mseccfg, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x70,0x74]
# CHECK-INST-ALIAS: csrr t2, mseccfg
# name
csrrs t1, mseccfg, zero
# uimm12
csrrs t2, 0x747, zero

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

# pmpcfg4
# name
# CHECK-INST: csrrs t1, pmpcfg4, zero
# CHECK-ENC: encoding: [0x73,0x23,0x40,0x3a]
# CHECK-INST-ALIAS: csrr t1, pmpcfg4
# uimm12
# CHECK-INST: csrrs t2, pmpcfg4, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x40,0x3a]
# CHECK-INST-ALIAS: csrr t2, pmpcfg4
# name
csrrs t1, pmpcfg4, zero
# uimm12
csrrs t2, 0x3A4, zero

# pmpcfg6
# name
# CHECK-INST: csrrs t1, pmpcfg6, zero
# CHECK-ENC: encoding: [0x73,0x23,0x60,0x3a]
# CHECK-INST-ALIAS: csrr t1, pmpcfg6
# uimm12
# CHECK-INST: csrrs t2, pmpcfg6, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x60,0x3a]
# CHECK-INST-ALIAS: csrr t2, pmpcfg6
# name
csrrs t1, pmpcfg6, zero
# uimm12
csrrs t2, 0x3A6, zero

# pmpcfg8
# name
# CHECK-INST: csrrs t1, pmpcfg8, zero
# CHECK-ENC: encoding: [0x73,0x23,0x80,0x3a]
# CHECK-INST-ALIAS: csrr t1, pmpcfg8
# uimm12
# CHECK-INST: csrrs t2, pmpcfg8, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x80,0x3a]
# CHECK-INST-ALIAS: csrr t2, pmpcfg8
# name
csrrs t1, pmpcfg8, zero
# uimm12
csrrs t2, 0x3A8, zero

# pmpcfg10
# name
# CHECK-INST: csrrs t1, pmpcfg10, zero
# CHECK-ENC: encoding: [0x73,0x23,0xa0,0x3a]
# CHECK-INST-ALIAS: csrr t1, pmpcfg10
# uimm12
# CHECK-INST: csrrs t2, pmpcfg10, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xa0,0x3a]
# CHECK-INST-ALIAS: csrr t2, pmpcfg10
# name
csrrs t1, pmpcfg10, zero
# uimm12
csrrs t2, 0x3AA, zero

# pmpcfg12
# name
# CHECK-INST: csrrs t1, pmpcfg12, zero
# CHECK-ENC: encoding: [0x73,0x23,0xc0,0x3a]
# CHECK-INST-ALIAS: csrr t1, pmpcfg12
# uimm12
# CHECK-INST: csrrs t2, pmpcfg12, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xc0,0x3a]
# CHECK-INST-ALIAS: csrr t2, pmpcfg12
# name
csrrs t1, pmpcfg12, zero
# uimm12
csrrs t2, 0x3AC, zero

# pmpcfg14
# name
# CHECK-INST: csrrs t1, pmpcfg14, zero
# CHECK-ENC: encoding: [0x73,0x23,0xe0,0x3a]
# CHECK-INST-ALIAS: csrr t1, pmpcfg14
# uimm12
# CHECK-INST: csrrs t2, pmpcfg14, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xe0,0x3a]
# CHECK-INST-ALIAS: csrr t2, pmpcfg14
# name
csrrs t1, pmpcfg14, zero
# uimm12
csrrs t2, 0x3AE, zero

# pmpaddr0
# name
# CHECK-INST: csrrs t1, pmpaddr0, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0x3b]
# CHECK-INST-ALIAS: csrr t1, pmpaddr0
# uimm12
# CHECK-INST: csrrs t2, pmpaddr0, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0x3b]
# CHECK-INST-ALIAS: csrr t2, pmpaddr0
# name
csrrs t1, pmpaddr0, zero
# uimm12
csrrs t2, 0x3B0, zero

# pmpaddr1
# name
# CHECK-INST: csrrs t1, pmpaddr1, zero
# CHECK-ENC: encoding: [0x73,0x23,0x10,0x3b]
# CHECK-INST-ALIAS: csrr t1, pmpaddr1
# uimm12
# CHECK-INST: csrrs t2, pmpaddr1, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x10,0x3b]
# CHECK-INST-ALIAS: csrr t2, pmpaddr1
# name
csrrs t1, pmpaddr1, zero
# uimm12
csrrs t2, 0x3B1, zero

# pmpaddr2
# name
# CHECK-INST: csrrs t1, pmpaddr2, zero
# CHECK-ENC: encoding: [0x73,0x23,0x20,0x3b]
# CHECK-INST-ALIAS: csrr t1, pmpaddr2
# uimm12
# CHECK-INST: csrrs t2, pmpaddr2, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x20,0x3b]
# CHECK-INST-ALIAS: csrr t2, pmpaddr2
# name
csrrs t1, pmpaddr2, zero
# uimm12
csrrs t2, 0x3B2, zero

# pmpaddr3
# name
# CHECK-INST: csrrs t1, pmpaddr3, zero
# CHECK-ENC: encoding: [0x73,0x23,0x30,0x3b]
# CHECK-INST-ALIAS: csrr t1, pmpaddr3
# uimm12
# CHECK-INST: csrrs t2, pmpaddr3, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x30,0x3b]
# CHECK-INST-ALIAS: csrr t2, pmpaddr3
# name
csrrs t1, pmpaddr3, zero
# uimm12
csrrs t2, 0x3B3, zero

# pmpaddr4
# name
# CHECK-INST: csrrs t1, pmpaddr4, zero
# CHECK-ENC: encoding: [0x73,0x23,0x40,0x3b]
# CHECK-INST-ALIAS: csrr t1, pmpaddr4
# uimm12
# CHECK-INST: csrrs t2, pmpaddr4, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x40,0x3b]
# CHECK-INST-ALIAS: csrr t2, pmpaddr4
# name
csrrs t1, pmpaddr4, zero
# uimm12
csrrs t2, 0x3B4, zero

# pmpaddr5
# name
# CHECK-INST: csrrs t1, pmpaddr5, zero
# CHECK-ENC: encoding: [0x73,0x23,0x50,0x3b]
# CHECK-INST-ALIAS: csrr t1, pmpaddr5
# uimm12
# CHECK-INST: csrrs t2, pmpaddr5, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x50,0x3b]
# CHECK-INST-ALIAS: csrr t2, pmpaddr5
# name
csrrs t1, pmpaddr5, zero
# uimm12
csrrs t2, 0x3B5, zero

# pmpaddr6
# name
# CHECK-INST: csrrs t1, pmpaddr6, zero
# CHECK-ENC: encoding: [0x73,0x23,0x60,0x3b]
# CHECK-INST-ALIAS: csrr t1, pmpaddr6
# uimm12
# CHECK-INST: csrrs t2, pmpaddr6, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x60,0x3b]
# CHECK-INST-ALIAS: csrr t2, pmpaddr6
# name
csrrs t1, pmpaddr6, zero
# uimm12
csrrs t2, 0x3B6, zero

# pmpaddr7
# name
# CHECK-INST: csrrs t1, pmpaddr7, zero
# CHECK-ENC: encoding: [0x73,0x23,0x70,0x3b]
# CHECK-INST-ALIAS: csrr t1, pmpaddr7
# uimm12
# CHECK-INST: csrrs t2, pmpaddr7, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x70,0x3b]
# CHECK-INST-ALIAS: csrr t2, pmpaddr7
# name
csrrs t1, pmpaddr7, zero
# uimm12
csrrs t2, 0x3B7, zero

# pmpaddr8
# name
# CHECK-INST: csrrs t1, pmpaddr8, zero
# CHECK-ENC: encoding: [0x73,0x23,0x80,0x3b]
# CHECK-INST-ALIAS: csrr t1, pmpaddr8
# uimm12
# CHECK-INST: csrrs t2, pmpaddr8, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x80,0x3b]
# CHECK-INST-ALIAS: csrr t2, pmpaddr8
# name
csrrs t1, pmpaddr8, zero
# uimm12
csrrs t2, 0x3B8, zero

# pmpaddr9
# name
# CHECK-INST: csrrs t1, pmpaddr9, zero
# CHECK-ENC: encoding: [0x73,0x23,0x90,0x3b]
# CHECK-INST-ALIAS: csrr t1, pmpaddr9
# uimm12
# CHECK-INST: csrrs t2, pmpaddr9, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x90,0x3b]
# CHECK-INST-ALIAS: csrr t2, pmpaddr9
# name
csrrs t1, pmpaddr9, zero
# uimm12
csrrs t2, 0x3B9, zero

# pmpaddr10
# name
# CHECK-INST: csrrs t1, pmpaddr10, zero
# CHECK-ENC: encoding: [0x73,0x23,0xa0,0x3b]
# CHECK-INST-ALIAS: csrr t1, pmpaddr10
# uimm12
# CHECK-INST: csrrs t2, pmpaddr10, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xa0,0x3b]
# CHECK-INST-ALIAS: csrr t2, pmpaddr10
# name
csrrs t1, pmpaddr10, zero
# uimm12
csrrs t2, 0x3BA, zero

# pmpaddr11
# name
# CHECK-INST: csrrs t1, pmpaddr11, zero
# CHECK-ENC: encoding: [0x73,0x23,0xb0,0x3b]
# CHECK-INST-ALIAS: csrr t1, pmpaddr11
# uimm12
# CHECK-INST: csrrs t2, pmpaddr11, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xb0,0x3b]
# CHECK-INST-ALIAS: csrr t2, pmpaddr11
# name
csrrs t1, pmpaddr11, zero
# uimm12
csrrs t2, 0x3BB, zero

# pmpaddr12
# name
# CHECK-INST: csrrs t1, pmpaddr12, zero
# CHECK-ENC: encoding: [0x73,0x23,0xc0,0x3b]
# CHECK-INST-ALIAS: csrr t1, pmpaddr12
# uimm12
# CHECK-INST: csrrs t2, pmpaddr12, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xc0,0x3b]
# CHECK-INST-ALIAS: csrr t2, pmpaddr12
# name
csrrs t1, pmpaddr12, zero
# uimm12
csrrs t2, 0x3BC, zero

# pmpaddr13
# name
# CHECK-INST: csrrs t1, pmpaddr13, zero
# CHECK-ENC: encoding: [0x73,0x23,0xd0,0x3b]
# CHECK-INST-ALIAS: csrr t1, pmpaddr13
# uimm12
# CHECK-INST: csrrs t2, pmpaddr13, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xd0,0x3b]
# CHECK-INST-ALIAS: csrr t2, pmpaddr13
# name
csrrs t1, pmpaddr13, zero
# uimm12
csrrs t2, 0x3BD, zero

# pmpaddr14
# name
# CHECK-INST: csrrs t1, pmpaddr14, zero
# CHECK-ENC: encoding: [0x73,0x23,0xe0,0x3b]
# CHECK-INST-ALIAS: csrr t1, pmpaddr14
# uimm12
# CHECK-INST: csrrs t2, pmpaddr14, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xe0,0x3b]
# CHECK-INST-ALIAS: csrr t2, pmpaddr14
# name
csrrs t1, pmpaddr14, zero
# uimm12
csrrs t2, 0x3BE, zero

# pmpaddr15
# name
# CHECK-INST: csrrs t1, pmpaddr15, zero
# CHECK-ENC: encoding: [0x73,0x23,0xf0,0x3b]
# CHECK-INST-ALIAS: csrr t1, pmpaddr15
# uimm12
# CHECK-INST: csrrs t2, pmpaddr15, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xf0,0x3b]
# CHECK-INST-ALIAS: csrr t2, pmpaddr15
# name
csrrs t1, pmpaddr15, zero
# uimm12
csrrs t2, 0x3BF, zero

# pmpaddr16
# name
# CHECK-INST: csrrs t1, pmpaddr16, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0x3c]
# CHECK-INST-ALIAS: csrr t1, pmpaddr16
# uimm12
# CHECK-INST: csrrs t2, pmpaddr16, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0x3c]
# CHECK-INST-ALIAS: csrr t2, pmpaddr16
# name
csrrs t1, pmpaddr16, zero
# uimm12
csrrs t2, 0X3C0, zero

# pmpaddr17
# name
# CHECK-INST: csrrs t1, pmpaddr17, zero
# CHECK-ENC: encoding: [0x73,0x23,0x10,0x3c]
# CHECK-INST-ALIAS: csrr t1, pmpaddr17
# uimm12
# CHECK-INST: csrrs t2, pmpaddr17, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x10,0x3c]
# CHECK-INST-ALIAS: csrr t2, pmpaddr17
# name
csrrs t1, pmpaddr17, zero
# uimm12
csrrs t2, 0X3C1, zero

# pmpaddr18
# name
# CHECK-INST: csrrs t1, pmpaddr18, zero
# CHECK-ENC: encoding: [0x73,0x23,0x20,0x3c]
# CHECK-INST-ALIAS: csrr t1, pmpaddr18
# uimm12
# CHECK-INST: csrrs t2, pmpaddr18, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x20,0x3c]
# CHECK-INST-ALIAS: csrr t2, pmpaddr18
# name
csrrs t1, pmpaddr18, zero
# uimm12
csrrs t2, 0X3C2, zero

# pmpaddr19
# name
# CHECK-INST: csrrs t1, pmpaddr19, zero
# CHECK-ENC: encoding: [0x73,0x23,0x30,0x3c]
# CHECK-INST-ALIAS: csrr t1, pmpaddr19
# uimm12
# CHECK-INST: csrrs t2, pmpaddr19, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x30,0x3c]
# CHECK-INST-ALIAS: csrr t2, pmpaddr19
# name
csrrs t1, pmpaddr19, zero
# uimm12
csrrs t2, 0X3C3, zero

# pmpaddr20
# name
# CHECK-INST: csrrs t1, pmpaddr20, zero
# CHECK-ENC: encoding: [0x73,0x23,0x40,0x3c]
# CHECK-INST-ALIAS: csrr t1, pmpaddr20
# uimm12
# CHECK-INST: csrrs t2, pmpaddr20, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x40,0x3c]
# CHECK-INST-ALIAS: csrr t2, pmpaddr20
# name
csrrs t1, pmpaddr20, zero
# uimm12
csrrs t2, 0X3C4, zero

# pmpaddr21
# name
# CHECK-INST: csrrs t1, pmpaddr21, zero
# CHECK-ENC: encoding: [0x73,0x23,0x50,0x3c]
# CHECK-INST-ALIAS: csrr t1, pmpaddr21
# uimm12
# CHECK-INST: csrrs t2, pmpaddr21, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x50,0x3c]
# CHECK-INST-ALIAS: csrr t2, pmpaddr21
# name
csrrs t1, pmpaddr21, zero
# uimm12
csrrs t2, 0X3C5, zero

# pmpaddr22
# name
# CHECK-INST: csrrs t1, pmpaddr22, zero
# CHECK-ENC: encoding: [0x73,0x23,0x60,0x3c]
# CHECK-INST-ALIAS: csrr t1, pmpaddr22
# uimm12
# CHECK-INST: csrrs t2, pmpaddr22, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x60,0x3c]
# CHECK-INST-ALIAS: csrr t2, pmpaddr22
# name
csrrs t1, pmpaddr22, zero
# uimm12
csrrs t2, 0X3C6, zero

# pmpaddr23
# name
# CHECK-INST: csrrs t1, pmpaddr23, zero
# CHECK-ENC: encoding: [0x73,0x23,0x70,0x3c]
# CHECK-INST-ALIAS: csrr t1, pmpaddr23
# uimm12
# CHECK-INST: csrrs t2, pmpaddr23, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x70,0x3c]
# CHECK-INST-ALIAS: csrr t2, pmpaddr23
# name
csrrs t1, pmpaddr23, zero
# uimm12
csrrs t2, 0X3C7, zero

# pmpaddr24
# name
# CHECK-INST: csrrs t1, pmpaddr24, zero
# CHECK-ENC: encoding: [0x73,0x23,0x80,0x3c]
# CHECK-INST-ALIAS: csrr t1, pmpaddr24
# uimm12
# CHECK-INST: csrrs t2, pmpaddr24, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x80,0x3c]
# CHECK-INST-ALIAS: csrr t2, pmpaddr24
# name
csrrs t1, pmpaddr24, zero
# uimm12
csrrs t2, 0X3C8, zero

# pmpaddr25
# name
# CHECK-INST: csrrs t1, pmpaddr25, zero
# CHECK-ENC: encoding: [0x73,0x23,0x90,0x3c]
# CHECK-INST-ALIAS: csrr t1, pmpaddr25
# uimm12
# CHECK-INST: csrrs t2, pmpaddr25, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x90,0x3c]
# CHECK-INST-ALIAS: csrr t2, pmpaddr25
# name
csrrs t1, pmpaddr25, zero
# uimm12
csrrs t2, 0X3C9, zero

# pmpaddr26
# name
# CHECK-INST: csrrs t1, pmpaddr26, zero
# CHECK-ENC: encoding: [0x73,0x23,0xa0,0x3c]
# CHECK-INST-ALIAS: csrr t1, pmpaddr26
# uimm12
# CHECK-INST: csrrs t2, pmpaddr26, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xa0,0x3c]
# CHECK-INST-ALIAS: csrr t2, pmpaddr26
# name
csrrs t1, pmpaddr26, zero
# uimm12
csrrs t2, 0X3CA, zero

# pmpaddr27
# name
# CHECK-INST: csrrs t1, pmpaddr27, zero
# CHECK-ENC: encoding: [0x73,0x23,0xb0,0x3c]
# CHECK-INST-ALIAS: csrr t1, pmpaddr27
# uimm12
# CHECK-INST: csrrs t2, pmpaddr27, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xb0,0x3c]
# CHECK-INST-ALIAS: csrr t2, pmpaddr27
# name
csrrs t1, pmpaddr27, zero
# uimm12
csrrs t2, 0X3CB, zero

# pmpaddr28
# name
# CHECK-INST: csrrs t1, pmpaddr28, zero
# CHECK-ENC: encoding: [0x73,0x23,0xc0,0x3c]
# CHECK-INST-ALIAS: csrr t1, pmpaddr28
# uimm12
# CHECK-INST: csrrs t2, pmpaddr28, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xc0,0x3c]
# CHECK-INST-ALIAS: csrr t2, pmpaddr28
# name
csrrs t1, pmpaddr28, zero
# uimm12
csrrs t2, 0X3CC, zero

# pmpaddr29
# name
# CHECK-INST: csrrs t1, pmpaddr29, zero
# CHECK-ENC: encoding: [0x73,0x23,0xd0,0x3c]
# CHECK-INST-ALIAS: csrr t1, pmpaddr29
# uimm12
# CHECK-INST: csrrs t2, pmpaddr29, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xd0,0x3c]
# CHECK-INST-ALIAS: csrr t2, pmpaddr29
# name
csrrs t1, pmpaddr29, zero
# uimm12
csrrs t2, 0X3CD, zero

# pmpaddr30
# name
# CHECK-INST: csrrs t1, pmpaddr30, zero
# CHECK-ENC: encoding: [0x73,0x23,0xe0,0x3c]
# CHECK-INST-ALIAS: csrr t1, pmpaddr30
# uimm12
# CHECK-INST: csrrs t2, pmpaddr30, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xe0,0x3c]
# CHECK-INST-ALIAS: csrr t2, pmpaddr30
# name
csrrs t1, pmpaddr30, zero
# uimm12
csrrs t2, 0X3CE, zero

# pmpaddr31
# name
# CHECK-INST: csrrs t1, pmpaddr31, zero
# CHECK-ENC: encoding: [0x73,0x23,0xf0,0x3c]
# CHECK-INST-ALIAS: csrr t1, pmpaddr31
# uimm12
# CHECK-INST: csrrs t2, pmpaddr31, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xf0,0x3c]
# CHECK-INST-ALIAS: csrr t2, pmpaddr31
# name
csrrs t1, pmpaddr31, zero
# uimm12
csrrs t2, 0X3CF, zero

# pmpaddr32
# name
# CHECK-INST: csrrs t1, pmpaddr32, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0x3d]
# CHECK-INST-ALIAS: csrr t1, pmpaddr32
# uimm12
# CHECK-INST: csrrs t2, pmpaddr32, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0x3d]
# CHECK-INST-ALIAS: csrr t2, pmpaddr32
# name
csrrs t1, pmpaddr32, zero
# uimm12
csrrs t2, 0X3D0, zero

# pmpaddr33
# name
# CHECK-INST: csrrs t1, pmpaddr33, zero
# CHECK-ENC: encoding: [0x73,0x23,0x10,0x3d]
# CHECK-INST-ALIAS: csrr t1, pmpaddr33
# uimm12
# CHECK-INST: csrrs t2, pmpaddr33, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x10,0x3d]
# CHECK-INST-ALIAS: csrr t2, pmpaddr33
# name
csrrs t1, pmpaddr33, zero
# uimm12
csrrs t2, 0X3D1, zero

# pmpaddr34
# name
# CHECK-INST: csrrs t1, pmpaddr34, zero
# CHECK-ENC: encoding: [0x73,0x23,0x20,0x3d]
# CHECK-INST-ALIAS: csrr t1, pmpaddr34
# uimm12
# CHECK-INST: csrrs t2, pmpaddr34, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x20,0x3d]
# CHECK-INST-ALIAS: csrr t2, pmpaddr34
# name
csrrs t1, pmpaddr34, zero
# uimm12
csrrs t2, 0X3D2, zero

# pmpaddr35
# name
# CHECK-INST: csrrs t1, pmpaddr35, zero
# CHECK-ENC: encoding: [0x73,0x23,0x30,0x3d]
# CHECK-INST-ALIAS: csrr t1, pmpaddr35
# uimm12
# CHECK-INST: csrrs t2, pmpaddr35, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x30,0x3d]
# CHECK-INST-ALIAS: csrr t2, pmpaddr35
# name
csrrs t1, pmpaddr35, zero
# uimm12
csrrs t2, 0X3D3, zero

# pmpaddr36
# name
# CHECK-INST: csrrs t1, pmpaddr36, zero
# CHECK-ENC: encoding: [0x73,0x23,0x40,0x3d]
# CHECK-INST-ALIAS: csrr t1, pmpaddr36
# uimm12
# CHECK-INST: csrrs t2, pmpaddr36, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x40,0x3d]
# CHECK-INST-ALIAS: csrr t2, pmpaddr36
# name
csrrs t1, pmpaddr36, zero
# uimm12
csrrs t2, 0X3D4, zero

# pmpaddr37
# name
# CHECK-INST: csrrs t1, pmpaddr37, zero
# CHECK-ENC: encoding: [0x73,0x23,0x50,0x3d]
# CHECK-INST-ALIAS: csrr t1, pmpaddr37
# uimm12
# CHECK-INST: csrrs t2, pmpaddr37, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x50,0x3d]
# CHECK-INST-ALIAS: csrr t2, pmpaddr37
# name
csrrs t1, pmpaddr37, zero
# uimm12
csrrs t2, 0X3D5, zero

# pmpaddr38
# name
# CHECK-INST: csrrs t1, pmpaddr38, zero
# CHECK-ENC: encoding: [0x73,0x23,0x60,0x3d]
# CHECK-INST-ALIAS: csrr t1, pmpaddr38
# uimm12
# CHECK-INST: csrrs t2, pmpaddr38, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x60,0x3d]
# CHECK-INST-ALIAS: csrr t2, pmpaddr38
# name
csrrs t1, pmpaddr38, zero
# uimm12
csrrs t2, 0X3D6, zero

# pmpaddr39
# name
# CHECK-INST: csrrs t1, pmpaddr39, zero
# CHECK-ENC: encoding: [0x73,0x23,0x70,0x3d]
# CHECK-INST-ALIAS: csrr t1, pmpaddr39
# uimm12
# CHECK-INST: csrrs t2, pmpaddr39, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x70,0x3d]
# CHECK-INST-ALIAS: csrr t2, pmpaddr39
# name
csrrs t1, pmpaddr39, zero
# uimm12
csrrs t2, 0X3D7, zero

# pmpaddr40
# name
# CHECK-INST: csrrs t1, pmpaddr40, zero
# CHECK-ENC: encoding: [0x73,0x23,0x80,0x3d]
# CHECK-INST-ALIAS: csrr t1, pmpaddr40
# uimm12
# CHECK-INST: csrrs t2, pmpaddr40, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x80,0x3d]
# CHECK-INST-ALIAS: csrr t2, pmpaddr40
# name
csrrs t1, pmpaddr40, zero
# uimm12
csrrs t2, 0X3D8, zero

# pmpaddr41
# name
# CHECK-INST: csrrs t1, pmpaddr41, zero
# CHECK-ENC: encoding: [0x73,0x23,0x90,0x3d]
# CHECK-INST-ALIAS: csrr t1, pmpaddr41
# uimm12
# CHECK-INST: csrrs t2, pmpaddr41, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x90,0x3d]
# CHECK-INST-ALIAS: csrr t2, pmpaddr41
# name
csrrs t1, pmpaddr41, zero
# uimm12
csrrs t2, 0X3D9, zero

# pmpaddr42
# name
# CHECK-INST: csrrs t1, pmpaddr42, zero
# CHECK-ENC: encoding: [0x73,0x23,0xa0,0x3d]
# CHECK-INST-ALIAS: csrr t1, pmpaddr42
# uimm12
# CHECK-INST: csrrs t2, pmpaddr42, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xa0,0x3d]
# CHECK-INST-ALIAS: csrr t2, pmpaddr42
# name
csrrs t1, pmpaddr42, zero
# uimm12
csrrs t2, 0X3DA, zero

# pmpaddr43
# name
# CHECK-INST: csrrs t1, pmpaddr43, zero
# CHECK-ENC: encoding: [0x73,0x23,0xb0,0x3d]
# CHECK-INST-ALIAS: csrr t1, pmpaddr43
# uimm12
# CHECK-INST: csrrs t2, pmpaddr43, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xb0,0x3d]
# CHECK-INST-ALIAS: csrr t2, pmpaddr43
# name
csrrs t1, pmpaddr43, zero
# uimm12
csrrs t2, 0X3DB, zero

# pmpaddr44
# name
# CHECK-INST: csrrs t1, pmpaddr44, zero
# CHECK-ENC: encoding: [0x73,0x23,0xc0,0x3d]
# CHECK-INST-ALIAS: csrr t1, pmpaddr44
# uimm12
# CHECK-INST: csrrs t2, pmpaddr44, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xc0,0x3d]
# CHECK-INST-ALIAS: csrr t2, pmpaddr44
# name
csrrs t1, pmpaddr44, zero
# uimm12
csrrs t2, 0X3DC, zero

# pmpaddr45
# name
# CHECK-INST: csrrs t1, pmpaddr45, zero
# CHECK-ENC: encoding: [0x73,0x23,0xd0,0x3d]
# CHECK-INST-ALIAS: csrr t1, pmpaddr45
# uimm12
# CHECK-INST: csrrs t2, pmpaddr45, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xd0,0x3d]
# CHECK-INST-ALIAS: csrr t2, pmpaddr45
# name
csrrs t1, pmpaddr45, zero
# uimm12
csrrs t2, 0X3DD, zero

# pmpaddr46
# name
# CHECK-INST: csrrs t1, pmpaddr46, zero
# CHECK-ENC: encoding: [0x73,0x23,0xe0,0x3d]
# CHECK-INST-ALIAS: csrr t1, pmpaddr46
# uimm12
# CHECK-INST: csrrs t2, pmpaddr46, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xe0,0x3d]
# CHECK-INST-ALIAS: csrr t2, pmpaddr46
# name
csrrs t1, pmpaddr46, zero
# uimm12
csrrs t2, 0X3DE, zero

# pmpaddr47
# name
# CHECK-INST: csrrs t1, pmpaddr47, zero
# CHECK-ENC: encoding: [0x73,0x23,0xf0,0x3d]
# CHECK-INST-ALIAS: csrr t1, pmpaddr47
# uimm12
# CHECK-INST: csrrs t2, pmpaddr47, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xf0,0x3d]
# CHECK-INST-ALIAS: csrr t2, pmpaddr47
# name
csrrs t1, pmpaddr47, zero
# uimm12
csrrs t2, 0X3DF, zero

# pmpaddr48
# name
# CHECK-INST: csrrs t1, pmpaddr48, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0x3e]
# CHECK-INST-ALIAS: csrr t1, pmpaddr48
# uimm12
# CHECK-INST: csrrs t2, pmpaddr48, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0x3e]
# CHECK-INST-ALIAS: csrr t2, pmpaddr48
# name
csrrs t1, pmpaddr48, zero
# uimm12
csrrs t2, 0X3E0, zero

# pmpaddr49
# name
# CHECK-INST: csrrs t1, pmpaddr49, zero
# CHECK-ENC: encoding: [0x73,0x23,0x10,0x3e]
# CHECK-INST-ALIAS: csrr t1, pmpaddr49
# uimm12
# CHECK-INST: csrrs t2, pmpaddr49, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x10,0x3e]
# CHECK-INST-ALIAS: csrr t2, pmpaddr49
# name
csrrs t1, pmpaddr49, zero
# uimm12
csrrs t2, 0X3E1, zero

# pmpaddr50
# name
# CHECK-INST: csrrs t1, pmpaddr50, zero
# CHECK-ENC: encoding: [0x73,0x23,0x20,0x3e]
# CHECK-INST-ALIAS: csrr t1, pmpaddr50
# uimm12
# CHECK-INST: csrrs t2, pmpaddr50, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x20,0x3e]
# CHECK-INST-ALIAS: csrr t2, pmpaddr50
# name
csrrs t1, pmpaddr50, zero
# uimm12
csrrs t2, 0X3E2, zero

# pmpaddr51
# name
# CHECK-INST: csrrs t1, pmpaddr51, zero
# CHECK-ENC: encoding: [0x73,0x23,0x30,0x3e]
# CHECK-INST-ALIAS: csrr t1, pmpaddr51
# uimm12
# CHECK-INST: csrrs t2, pmpaddr51, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x30,0x3e]
# CHECK-INST-ALIAS: csrr t2, pmpaddr51
# name
csrrs t1, pmpaddr51, zero
# uimm12
csrrs t2, 0X3E3, zero

# pmpaddr52
# name
# CHECK-INST: csrrs t1, pmpaddr52, zero
# CHECK-ENC: encoding: [0x73,0x23,0x40,0x3e]
# CHECK-INST-ALIAS: csrr t1, pmpaddr52
# uimm12
# CHECK-INST: csrrs t2, pmpaddr52, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x40,0x3e]
# CHECK-INST-ALIAS: csrr t2, pmpaddr52
# name
csrrs t1, pmpaddr52, zero
# uimm12
csrrs t2, 0X3E4, zero

# pmpaddr53
# name
# CHECK-INST: csrrs t1, pmpaddr53, zero
# CHECK-ENC: encoding: [0x73,0x23,0x50,0x3e]
# CHECK-INST-ALIAS: csrr t1, pmpaddr53
# uimm12
# CHECK-INST: csrrs t2, pmpaddr53, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x50,0x3e]
# CHECK-INST-ALIAS: csrr t2, pmpaddr53
# name
csrrs t1, pmpaddr53, zero
# uimm12
csrrs t2, 0X3E5, zero

# pmpaddr54
# name
# CHECK-INST: csrrs t1, pmpaddr54, zero
# CHECK-ENC: encoding: [0x73,0x23,0x60,0x3e]
# CHECK-INST-ALIAS: csrr t1, pmpaddr54
# uimm12
# CHECK-INST: csrrs t2, pmpaddr54, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x60,0x3e]
# CHECK-INST-ALIAS: csrr t2, pmpaddr54
# name
csrrs t1, pmpaddr54, zero
# uimm12
csrrs t2, 0X3E6, zero

# pmpaddr55
# name
# CHECK-INST: csrrs t1, pmpaddr55, zero
# CHECK-ENC: encoding: [0x73,0x23,0x70,0x3e]
# CHECK-INST-ALIAS: csrr t1, pmpaddr55
# uimm12
# CHECK-INST: csrrs t2, pmpaddr55, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x70,0x3e]
# CHECK-INST-ALIAS: csrr t2, pmpaddr55
# name
csrrs t1, pmpaddr55, zero
# uimm12
csrrs t2, 0X3E7, zero

# pmpaddr56
# name
# CHECK-INST: csrrs t1, pmpaddr56, zero
# CHECK-ENC: encoding: [0x73,0x23,0x80,0x3e]
# CHECK-INST-ALIAS: csrr t1, pmpaddr56
# uimm12
# CHECK-INST: csrrs t2, pmpaddr56, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x80,0x3e]
# CHECK-INST-ALIAS: csrr t2, pmpaddr56
# name
csrrs t1, pmpaddr56, zero
# uimm12
csrrs t2, 0X3E8, zero

# pmpaddr57
# name
# CHECK-INST: csrrs t1, pmpaddr57, zero
# CHECK-ENC: encoding: [0x73,0x23,0x90,0x3e]
# CHECK-INST-ALIAS: csrr t1, pmpaddr57
# uimm12
# CHECK-INST: csrrs t2, pmpaddr57, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x90,0x3e]
# CHECK-INST-ALIAS: csrr t2, pmpaddr57
# name
csrrs t1, pmpaddr57, zero
# uimm12
csrrs t2, 0X3E9, zero

# pmpaddr58
# name
# CHECK-INST: csrrs t1, pmpaddr58, zero
# CHECK-ENC: encoding: [0x73,0x23,0xa0,0x3e]
# CHECK-INST-ALIAS: csrr t1, pmpaddr58
# uimm12
# CHECK-INST: csrrs t2, pmpaddr58, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xa0,0x3e]
# CHECK-INST-ALIAS: csrr t2, pmpaddr58
# name
csrrs t1, pmpaddr58, zero
# uimm12
csrrs t2, 0X3EA, zero

# pmpaddr59
# name
# CHECK-INST: csrrs t1, pmpaddr59, zero
# CHECK-ENC: encoding: [0x73,0x23,0xb0,0x3e]
# CHECK-INST-ALIAS: csrr t1, pmpaddr59
# uimm12
# CHECK-INST: csrrs t2, pmpaddr59, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xb0,0x3e]
# CHECK-INST-ALIAS: csrr t2, pmpaddr59
# name
csrrs t1, pmpaddr59, zero
# uimm12
csrrs t2, 0X3EB, zero

# pmpaddr60
# name
# CHECK-INST: csrrs t1, pmpaddr60, zero
# CHECK-ENC: encoding: [0x73,0x23,0xc0,0x3e]
# CHECK-INST-ALIAS: csrr t1, pmpaddr60
# uimm12
# CHECK-INST: csrrs t2, pmpaddr60, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xc0,0x3e]
# CHECK-INST-ALIAS: csrr t2, pmpaddr60
# name
csrrs t1, pmpaddr60, zero
# uimm12
csrrs t2, 0X3EC, zero

# pmpaddr61
# name
# CHECK-INST: csrrs t1, pmpaddr61, zero
# CHECK-ENC: encoding: [0x73,0x23,0xd0,0x3e]
# CHECK-INST-ALIAS: csrr t1, pmpaddr61
# uimm12
# CHECK-INST: csrrs t2, pmpaddr61, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xd0,0x3e]
# CHECK-INST-ALIAS: csrr t2, pmpaddr61
# name
csrrs t1, pmpaddr61, zero
# uimm12
csrrs t2, 0X3ED, zero

# pmpaddr62
# name
# CHECK-INST: csrrs t1, pmpaddr62, zero
# CHECK-ENC: encoding: [0x73,0x23,0xe0,0x3e]
# CHECK-INST-ALIAS: csrr t1, pmpaddr62
# uimm12
# CHECK-INST: csrrs t2, pmpaddr62, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xe0,0x3e]
# CHECK-INST-ALIAS: csrr t2, pmpaddr62
# name
csrrs t1, pmpaddr62, zero
# uimm12
csrrs t2, 0X3EE, zero

# pmpaddr63
# name
# CHECK-INST: csrrs t1, pmpaddr63, zero
# CHECK-ENC: encoding: [0x73,0x23,0xf0,0x3e]
# CHECK-INST-ALIAS: csrr t1, pmpaddr63
# uimm12
# CHECK-INST: csrrs t2, pmpaddr63, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xf0,0x3e]
# CHECK-INST-ALIAS: csrr t2, pmpaddr63
# name
csrrs t1, pmpaddr63, zero
# uimm12
csrrs t2, 0X3EF, zero

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

# tdata3
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

# mcontext
# name
# CHECK-INST: csrrs t1, mcontext, zero
# CHECK-ENC: encoding: [0x73,0x23,0x80,0x7a]
# CHECK-INST-ALIAS: csrr t1, mcontext
# uimm12
# CHECK-INST: csrrs t2, mcontext, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x80,0x7a]
# CHECK-INST-ALIAS: csrr t2, mcontext
# name
csrrs t1, mcontext, zero
# uimm12
csrrs t2, 0x7A8, zero

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

# dscratch0
# name
# CHECK-INST: csrrs t1, dscratch0, zero
# CHECK-ENC: encoding: [0x73,0x23,0x20,0x7b]
# CHECK-INST-ALIAS: csrr t1, dscratch0
# uimm12
# CHECK-INST: csrrs t2, dscratch0, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x20,0x7b]
# CHECK-INST-ALIAS: csrr t2, dscratch0
# name
csrrs t1, dscratch0, zero
# uimm12
csrrs t2, 0x7B2, zero

# dscratch
# name
# CHECK-INST: csrrs t1, dscratch0, zero
# CHECK-ENC: encoding: [0x73,0x23,0x20,0x7b]
# CHECK-INST-ALIAS: csrr t1, dscratch0
# uimm12
# CHECK-INST: csrrs t2, dscratch0, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x20,0x7b]
# CHECK-INST-ALIAS: csrr t2, dscratch0
# name
csrrs t1, dscratch, zero
# uimm12
csrrs t2, 0x7B2, zero

# dscratch1
# name
# CHECK-INST: csrrs t1, dscratch1, zero
# CHECK-ENC: encoding: [0x73,0x23,0x30,0x7b]
# CHECK-INST-ALIAS: csrr t1, dscratch1
# uimm12
# CHECK-INST: csrrs t2, dscratch1, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x30,0x7b]
# CHECK-INST-ALIAS: csrr t2, dscratch1
# name
csrrs t1, dscratch1, zero
# uimm12
csrrs t2, 0x7B3, zero

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
# mcountinhibit
# name
# CHECK-INST: csrrs t1, mcountinhibit, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x00,0x32]
# CHECK-INST-ALIAS: csrr t1, mcountinhibit
# uimm12
# CHECK-INST: csrrs t2, mcountinhibit, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x00,0x32]
# CHECK-INST-ALIAS: csrr t2, mcountinhibit
# name
csrrs t1, mcountinhibit, zero
# uimm12
csrrs t2, 0x320, zero

# mucounteren
# name
# CHECK-INST: csrrs t1, mcountinhibit, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x00,0x32]
# CHECK-INST-ALIAS: csrr t1, mcountinhibit
# uimm12
# CHECK-INST: csrrs t2, mcountinhibit, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x00,0x32]
# CHECK-INST-ALIAS: csrr t2, mcountinhibit
# name
csrrs t1, mucounteren, zero
# uimm12
csrrs t2, 0x320, zero

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

#########################################
# State Enable Extension (Smstateen)
#########################################

# mstateen0
# name
# CHECK-INST: csrrs t1, mstateen0, zero
# CHECK-ENC: encoding: [0x73,0x23,0xc0,0x30]
# CHECK-INST-ALIAS: csrr t1, mstateen0
# uimm12
# CHECK-INST: csrrs t2, mstateen0, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xc0,0x30]
# CHECK-INST-ALIAS: csrr t2, mstateen0
# name
csrrs t1, mstateen0, zero
# uimm12
csrrs t2, 0x30C, zero

# mstateen1
# name
# CHECK-INST: csrrs t1, mstateen1, zero
# CHECK-ENC: encoding: [0x73,0x23,0xd0,0x30]
# CHECK-INST-ALIAS: csrr t1, mstateen1
# uimm12
# CHECK-INST: csrrs t2, mstateen1, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xd0,0x30]
# CHECK-INST-ALIAS: csrr t2, mstateen1
# name
csrrs t1, mstateen1, zero
# uimm12
csrrs t2, 0x30D, zero

# mstateen2
# name
# CHECK-INST: csrrs t1, mstateen2, zero
# CHECK-ENC: encoding: [0x73,0x23,0xe0,0x30]
# CHECK-INST-ALIAS: csrr t1, mstateen2
# uimm12
# CHECK-INST: csrrs t2, mstateen2, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xe0,0x30]
# CHECK-INST-ALIAS: csrr t2, mstateen2
# name
csrrs t1, mstateen2, zero
# uimm12
csrrs t2, 0x30E, zero

# mstateen3
# name
# CHECK-INST: csrrs t1, mstateen3, zero
# CHECK-ENC: encoding: [0x73,0x23,0xf0,0x30]
# CHECK-INST-ALIAS: csrr t1, mstateen3
# uimm12
# CHECK-INST: csrrs t2, mstateen3, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xf0,0x30]
# CHECK-INST-ALIAS: csrr t2, mstateen3
# name
csrrs t1, mstateen3, zero
# uimm12
csrrs t2, 0x30F, zero

#########################################
# Advanced Interrupt Architecture (Smaia and Ssaia)
#########################################

# miselect
# name
# CHECK-INST: csrrs t1, miselect, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0x35]
# CHECK-INST-ALIAS: csrr t1, miselect
# uimm12
# CHECK-INST: csrrs t2, miselect, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0x35]
# CHECK-INST-ALIAS: csrr t2, miselect
# name
csrrs t1, miselect, zero
# uimm12
csrrs t2, 0x350, zero

# mireg
# name
# CHECK-INST: csrrs t1, mireg, zero
# CHECK-ENC: encoding: [0x73,0x23,0x10,0x35]
# CHECK-INST-ALIAS: csrr t1, mireg
# uimm12
# CHECK-INST: csrrs t2, mireg, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x10,0x35]
# CHECK-INST-ALIAS: csrr t2, mireg
# name
csrrs t1, mireg, zero
# uimm12
csrrs t2, 0x351, zero

# mtopei
# name
# CHECK-INST: csrrs t1, mtopei, zero
# CHECK-ENC: encoding: [0x73,0x23,0xc0,0x35]
# CHECK-INST-ALIAS: csrr t1, mtopei
# uimm12
# CHECK-INST: csrrs t2, mtopei, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xc0,0x35]
# CHECK-INST-ALIAS: csrr t2, mtopei
# name
csrrs t1, mtopei, zero
# uimm12
csrrs t2, 0x35C, zero

# mtopi
# name
# CHECK-INST: csrrs t1, mtopi, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0xfb]
# CHECK-INST-ALIAS: csrr t1, mtopi
# uimm12
# CHECK-INST: csrrs t2, mtopi, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0xfb]
# CHECK-INST-ALIAS: csrr t2, mtopi
# name
csrrs t1, mtopi, zero
# uimm12
csrrs t2, 0xFB0, zero

# mvien
# name
# CHECK-INST: csrrs t1, mvien, zero
# CHECK-ENC: encoding: [0x73,0x23,0x80,0x30]
# CHECK-INST-ALIAS: csrr t1, mvien
# uimm12
# CHECK-INST: csrrs t2, mvien, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x80,0x30]
# CHECK-INST-ALIAS: csrr t2, mvien
# name
csrrs t1, mvien, zero
# uimm12
csrrs t2, 0x308, zero

# mvip
# name
# CHECK-INST: csrrs t1, mvip, zero
# CHECK-ENC: encoding: [0x73,0x23,0x90,0x30]
# CHECK-INST-ALIAS: csrr t1, mvip
# uimm12
# CHECK-INST: csrrs t2, mvip, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x90,0x30]
# CHECK-INST-ALIAS: csrr t2, mvip
# name
csrrs t1, mvip, zero
# uimm12
csrrs t2, 0x309, zero
