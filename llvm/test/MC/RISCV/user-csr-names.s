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
# User Trap Setup
##################################

# ustatus
# name
# CHECK-INST: csrrs t1, ustatus, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x00,0x00]
# CHECK-INST-ALIAS: csrr t1, ustatus
# uimm12
# CHECK-INST: csrrs t2, ustatus, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x00,0x00]
# CHECK-INST-ALIAS: csrr t2, ustatus
csrrs t1, ustatus, zero
# uimm12
csrrs t2, 0x000, zero

# uie
# name
# CHECK-INST: csrrs t1, uie, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x40,0x00]
# CHECK-INST-ALIAS: csrr t1, uie
# uimm12
# CHECK-INST: csrrs t2, uie, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x40,0x00]
# CHECK-INST-ALIAS: csrr t2, uie
# name
csrrs t1, uie, zero
# uimm12
csrrs t2, 0x004, zero

# utvec
# name
# CHECK-INST: csrrs t1, utvec, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x50,0x00]
# CHECK-INST-ALIAS: csrr t1, utvec
# uimm12
# CHECK-INST: csrrs t2, utvec, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x50,0x00]
# CHECK-INST-ALIAS: csrr t2, utvec
# name
csrrs t1, utvec, zero
# uimm12
csrrs t2, 0x005, zero

##################################
# User Trap Handling
##################################

# uscratch
# name
# CHECK-INST: csrrs t1, uscratch, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x00,0x04]
# CHECK-INST-ALIAS: csrr t1, uscratch
# uimm12
# CHECK-INST: csrrs t2, uscratch, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x00,0x04]
# CHECK-INST-ALIAS: csrr t2, uscratch
# name
csrrs t1, uscratch, zero
# uimm12
csrrs t2, 0x040, zero

# uepc
# name
# CHECK-INST: csrrs t1, uepc, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x10,0x04]
# CHECK-INST-ALIAS: csrr t1, uepc
# uimm12
# CHECK-INST: csrrs t2, uepc, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x10,0x04]
# CHECK-INST-ALIAS: csrr t2, uepc
# name
csrrs t1, uepc, zero
# uimm12
csrrs t2, 0x041, zero

# ucause
# name
# CHECK-INST: csrrs t1, ucause, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x20,0x04]
# CHECK-INST-ALIAS: csrr t1, ucause
# uimm12
# CHECK-INST: csrrs t2, ucause, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x20,0x04]
# CHECK-INST-ALIAS: csrr t2, ucause
# name
csrrs t1, ucause, zero
# uimm12
csrrs t2, 0x042, zero

# utval
# name
# CHECK-INST: csrrs t1, utval, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x30,0x04]
# CHECK-INST-ALIAS: csrr t1, utval
# uimm12
# CHECK-INST: csrrs t2, utval, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x30,0x04]
# CHECK-INST-ALIAS: csrr t2, utval
# name
csrrs t1, utval, zero
# uimm12
csrrs t2, 0x043, zero

# uip
# name
# CHECK-INST: csrrs t1, uip, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x40,0x04]
# CHECK-INST-ALIAS: csrr t1, uip
# uimm12
# CHECK-INST: csrrs t2, uip, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x40,0x04]
# CHECK-INST-ALIAS: csrr t2, uip
#name
csrrs t1, uip, zero
# uimm12
csrrs t2, 0x044, zero

##################################
# User Floating Pont CSRs
##################################
# Tests in rvf-user-mode-csr.s

##################################
# User Counter and Timers
##################################

# cycle
# name
# CHECK-INST: csrrs t1, cycle, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x00,0xc0]
# CHECK-INST-ALIAS: rdcycle t1
# uimm12
# CHECK-INST: csrrs t2, cycle, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x00,0xc0]
# CHECK-INST-ALIAS: rdcycle t2
# name
csrrs t1, cycle, zero
# uimm12
csrrs t2, 0xC00, zero

# time
# name
# CHECK-INST: csrrs t1, time, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x10,0xc0]
# CHECK-INST-ALIAS: rdtime t1
# uimm12
# CHECK-INST: csrrs t2, time, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x10,0xc0]
# CHECK-INST-ALIAS: rdtime t2
# name
csrrs t1, time, zero
# uimm12
csrrs t2, 0xC01, zero

# instret
# name
# CHECK-INST: csrrs t1, instret, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x20,0xc0]
# CHECK-INST-ALIAS: rdinstret t1
# uimm12
# CHECK-INST: csrrs t2, instret, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x20,0xc0]
# CHECK-INST-ALIAS: rdinstret t2
# name
csrrs t1, instret, zero
# uimm12
csrrs t2, 0xC02, zero

# hpmcounter3
# name
# CHECK-INST: csrrs t1, hpmcounter3, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x30,0xc0]
# CHECK-INST-ALIAS: csrr t1, hpmcounter3
# uimm12
# CHECK-INST: csrrs t2, hpmcounter3, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x30,0xc0]
# CHECK-INST-ALIAS: csrr t2, hpmcounter3
# name
csrrs t1, hpmcounter3, zero
# uimm12
csrrs t2, 0xC03, zero

# hpmcounter4
# name
# CHECK-INST: csrrs t1, hpmcounter4, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x40,0xc0]
# CHECK-INST-ALIAS: csrr t1, hpmcounter4
# uimm12
# CHECK-INST: csrrs t2, hpmcounter4, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x40,0xc0]
# CHECK-INST-ALIAS: csrr t2, hpmcounter4
# name
csrrs t1, hpmcounter4, zero
# uimm12
csrrs t2, 0xC04, zero

# hpmcounter5
# name
# CHECK-INST: csrrs t1, hpmcounter5, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x50,0xc0]
# CHECK-INST-ALIAS: csrr t1, hpmcounter5
# uimm12
# CHECK-INST: csrrs t2, hpmcounter5, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x50,0xc0]
# CHECK-INST-ALIAS: csrr t2, hpmcounter5
# name
csrrs t1, hpmcounter5, zero
# uimm12
csrrs t2, 0xC05, zero

# hpmcounter6
# name
# CHECK-INST: csrrs t1, hpmcounter6, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x60,0xc0]
# CHECK-INST-ALIAS: csrr t1, hpmcounter6
# uimm12
# CHECK-INST: csrrs t2, hpmcounter6, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x60,0xc0]
# CHECK-INST-ALIAS: csrr t2, hpmcounter6
# name
csrrs t1, hpmcounter6, zero
# uimm12
csrrs t2, 0xC06, zero

# hpmcounter7
# name
# CHECK-INST: csrrs t1, hpmcounter7, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x70,0xc0]
# CHECK-INST-ALIAS: csrr t1, hpmcounter7
# uimm12
# CHECK-INST: csrrs t2, hpmcounter7, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x70,0xc0]
# CHECK-INST-ALIAS: csrr t2, hpmcounter7
# name
csrrs t1, hpmcounter7, zero
# uimm12
csrrs t2, 0xC07, zero

# hpmcounter8
# name
# CHECK-INST: csrrs t1, hpmcounter8, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x80,0xc0]
# CHECK-INST-ALIAS: csrr t1, hpmcounter8
# uimm12
# CHECK-INST: csrrs t2, hpmcounter8, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x80,0xc0]
# CHECK-INST-ALIAS: csrr t2, hpmcounter8
# name
csrrs t1, hpmcounter8, zero
# uimm12
csrrs t2, 0xC08, zero

# hpmcounter9
# name
# CHECK-INST: csrrs t1, hpmcounter9, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x90,0xc0]
# CHECK-INST-ALIAS: csrr t1, hpmcounter9
# uimm12
# CHECK-INST: csrrs t2, hpmcounter9, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x90,0xc0]
# CHECK-INST-ALIAS: csrr t2, hpmcounter9
# name
csrrs t1, hpmcounter9, zero
# uimm12
csrrs t2, 0xC09, zero

# hpmcounter10
# name
# CHECK-INST: csrrs t1, hpmcounter10, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xa0,0xc0]
# CHECK-INST-ALIAS: csrr t1, hpmcounter10
# uimm12
# CHECK-INST: csrrs t2, hpmcounter10, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xa0,0xc0]
# CHECK-INST-ALIAS: csrr t2, hpmcounter10
# name
csrrs t1, hpmcounter10, zero
# uimm12
csrrs t2, 0xC0A, zero

# hpmcounter11
# name
# CHECK-INST: csrrs t1, hpmcounter11, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xb0,0xc0]
# CHECK-INST-ALIAS: csrr t1, hpmcounter11
# uimm12
# CHECK-INST: csrrs t2, hpmcounter11, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xb0,0xc0]
# CHECK-INST-ALIAS: csrr t2, hpmcounter11
# name
csrrs t1, hpmcounter11, zero
# uimm12
csrrs t2, 0xC0B, zero

# hpmcounter12
# name
# CHECK-INST: csrrs t1, hpmcounter12, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xc0,0xc0]
# CHECK-INST-ALIAS: csrr t1, hpmcounter12
# uimm12
# CHECK-INST: csrrs t2, hpmcounter12, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xc0,0xc0]
# CHECK-INST-ALIAS: csrr t2, hpmcounter12
# name
csrrs t1, hpmcounter12, zero
# uimm12
csrrs t2, 0xC0C, zero

# hpmcounter13
# name
# CHECK-INST: csrrs t1, hpmcounter13, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xd0,0xc0]
# CHECK-INST-ALIAS: csrr t1, hpmcounter13
# uimm12
# CHECK-INST: csrrs t2, hpmcounter13, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xd0,0xc0]
# CHECK-INST-ALIAS: csrr t2, hpmcounter13
# name
csrrs t1, hpmcounter13, zero
# uimm12
csrrs t2, 0xC0D, zero

# hpmcounter14
# name
# CHECK-INST: csrrs t1, hpmcounter14, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xe0,0xc0]
# CHECK-INST-ALIAS: csrr t1, hpmcounter14
# uimm12
# CHECK-INST: csrrs t2, hpmcounter14, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xe0,0xc0]
# CHECK-INST-ALIAS: csrr t2, hpmcounter14
# name
csrrs t1, hpmcounter14, zero
# uimm12
csrrs t2, 0xC0E, zero

# hpmcounter15
# name
# CHECK-INST: csrrs t1, hpmcounter15, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xf0,0xc0]
# CHECK-INST-ALIAS: csrr t1, hpmcounter15
# uimm12
# CHECK-INST: csrrs t2, hpmcounter15, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xf0,0xc0]
# CHECK-INST-ALIAS: csrr t2, hpmcounter15
# name
csrrs t1, hpmcounter15, zero
# uimm12
csrrs t2, 0xC0F, zero

# hpmcounter16
# name
# CHECK-INST: csrrs t1, hpmcounter16, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x00,0xc1]
# CHECK-INST-ALIAS: csrr t1, hpmcounter16
# uimm12
# CHECK-INST: csrrs t2, hpmcounter16, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x00,0xc1]
# CHECK-INST-ALIAS: csrr t2, hpmcounter16
# name
csrrs t1, hpmcounter16, zero
# uimm12
csrrs t2, 0xC10, zero

# hpmcounter17
# name
# CHECK-INST: csrrs t1, hpmcounter17, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x10,0xc1]
# CHECK-INST-ALIAS: csrr t1, hpmcounter17
# uimm12
# CHECK-INST: csrrs t2, hpmcounter17, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x10,0xc1]
# CHECK-INST-ALIAS: csrr t2, hpmcounter17
# name
csrrs t1, hpmcounter17, zero
# uimm12
csrrs t2, 0xC11, zero

# hpmcounter18
# name
# CHECK-INST: csrrs t1, hpmcounter18, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x20,0xc1]
# CHECK-INST-ALIAS: csrr t1, hpmcounter18
# uimm12
# CHECK-INST: csrrs t2, hpmcounter18, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x20,0xc1]
# CHECK-INST-ALIAS: csrr t2, hpmcounter18
# name
csrrs t1, hpmcounter18, zero
# uimm12
csrrs t2, 0xC12, zero

# hpmcounter19
# name
# CHECK-INST: csrrs t1, hpmcounter19, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x30,0xc1]
# CHECK-INST-ALIAS: csrr t1, hpmcounter19
# uimm12
# CHECK-INST: csrrs t2, hpmcounter19, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x30,0xc1]
# CHECK-INST-ALIAS: csrr t2, hpmcounter19
# name
csrrs t1, hpmcounter19, zero
# uimm12
csrrs t2, 0xC13, zero

# hpmcounter20
# name
# CHECK-INST: csrrs t1, hpmcounter20, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x40,0xc1]
# CHECK-INST-ALIAS: csrr t1, hpmcounter20
# uimm12
# CHECK-INST: csrrs t2, hpmcounter20, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x40,0xc1]
# CHECK-INST-ALIAS: csrr t2, hpmcounter20
# name
csrrs t1, hpmcounter20, zero
# uimm12
csrrs t2, 0xC14, zero

# hpmcounter21
# name
# CHECK-INST: csrrs t1, hpmcounter21, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x50,0xc1]
# CHECK-INST-ALIAS: csrr t1, hpmcounter21
# uimm12
# CHECK-INST: csrrs t2, hpmcounter21, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x50,0xc1]
# CHECK-INST-ALIAS: csrr t2, hpmcounter21
# name
csrrs t1, hpmcounter21, zero
# uimm12
csrrs t2, 0xC15, zero

# hpmcounter22
# name
# CHECK-INST: csrrs t1, hpmcounter22, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x60,0xc1]
# CHECK-INST-ALIAS: csrr t1, hpmcounter22
# uimm12
# CHECK-INST: csrrs t2, hpmcounter22, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x60,0xc1]
# CHECK-INST-ALIAS: csrr t2, hpmcounter22
# name
csrrs t1, hpmcounter22, zero
# uimm12
csrrs t2, 0xC16, zero

# hpmcounter23
# name
# CHECK-INST: csrrs t1, hpmcounter23, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x70,0xc1]
# CHECK-INST-ALIAS: csrr t1, hpmcounter23
# uimm12
# CHECK-INST: csrrs t2, hpmcounter23, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x70,0xc1]
# CHECK-INST-ALIAS: csrr t2, hpmcounter23
# name
csrrs t1, hpmcounter23, zero
# uimm12
csrrs t2, 0xC17, zero

# hpmcounter24
# name
# CHECK-INST: csrrs t1, hpmcounter24, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x80,0xc1]
# CHECK-INST-ALIAS: csrr t1, hpmcounter24
# uimm12
# CHECK-INST: csrrs t2, hpmcounter24, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x80,0xc1]
# CHECK-INST-ALIAS: csrr t2, hpmcounter24
# name
csrrs t1, hpmcounter24, zero
# uimm12
csrrs t2, 0xC18, zero

# hpmcounter25
# name
# CHECK-INST: csrrs t1, hpmcounter25, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x90,0xc1]
# CHECK-INST-ALIAS: csrr t1, hpmcounter25
# uimm12
# CHECK-INST: csrrs t2, hpmcounter25, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x90,0xc1]
# CHECK-INST-ALIAS: csrr t2, hpmcounter25
# name
csrrs t1, hpmcounter25, zero
# uimm12
csrrs t2, 0xC19, zero

# hpmcounter26
# name
# CHECK-INST: csrrs t1, hpmcounter26, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xa0,0xc1]
# CHECK-INST-ALIAS: csrr t1, hpmcounter26
# uimm12
# CHECK-INST: csrrs t2, hpmcounter26, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xa0,0xc1]
# CHECK-INST-ALIAS: csrr t2, hpmcounter26
# name
csrrs t1, hpmcounter26, zero
# uimm12
csrrs t2, 0xC1A, zero

# hpmcounter27
# name
# CHECK-INST: csrrs t1, hpmcounter27, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xb0,0xc1]
# CHECK-INST-ALIAS: csrr t1, hpmcounter27
# uimm12
# CHECK-INST: csrrs t2, hpmcounter27, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xb0,0xc1]
# CHECK-INST-ALIAS: csrr t2, hpmcounter27
# name
csrrs t1, hpmcounter27, zero
# uimm12
csrrs t2, 0xC1B, zero

# hpmcounter28
# name
# CHECK-INST: csrrs t1, hpmcounter28, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xc0,0xc1]
# CHECK-INST-ALIAS: csrr t1, hpmcounter28
# uimm12
# CHECK-INST: csrrs t2, hpmcounter28, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xc0,0xc1]
# CHECK-INST-ALIAS: csrr t2, hpmcounter28
# name
csrrs t1, hpmcounter28, zero
# uimm12
csrrs t2, 0xC1C, zero

# hpmcounter29
# name
# CHECK-INST: csrrs t1, hpmcounter29, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xd0,0xc1]
# CHECK-INST-ALIAS: csrr t1, hpmcounter29
# uimm12
# CHECK-INST: csrrs t2, hpmcounter29, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xd0,0xc1]
# CHECK-INST-ALIAS: csrr t2, hpmcounter29
# name
csrrs t1, hpmcounter29, zero
# uimm12
csrrs t2, 0xC1D, zero

# hpmcounter30
# name
# CHECK-INST: csrrs t1, hpmcounter30, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xe0,0xc1]
# CHECK-INST-ALIAS: csrr t1, hpmcounter30
# uimm12
# CHECK-INST: csrrs t2, hpmcounter30, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xe0,0xc1]
# CHECK-INST-ALIAS: csrr t2, hpmcounter30
# name
csrrs t1, hpmcounter30, zero
# uimm12
csrrs t2, 0xC1E, zero

# hpmcounter31
# name
# CHECK-INST: csrrs t1, hpmcounter31, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xf0,0xc1]
# CHECK-INST-ALIAS: csrr t1, hpmcounter31
# uimm12
# CHECK-INST: csrrs t2, hpmcounter31, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xf0,0xc1]
# CHECK-INST-ALIAS: csrr t2, hpmcounter31
# name
csrrs t1, hpmcounter31, zero
# uimm12
csrrs t2, 0xC1F, zero
