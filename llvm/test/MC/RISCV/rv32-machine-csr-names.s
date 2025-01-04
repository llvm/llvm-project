# RUN: llvm-mc %s -triple=riscv32 -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s

######################################
# Machine Trap Setup
######################################

# mstatush
# name
# CHECK-INST: csrrs t1, mstatush, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0x31]
# CHECK-INST-ALIAS: csrr t1, mstatush
# uimm12
# CHECK-INST: csrrs t2, mstatush, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0x31]
# CHECK-INST-ALIAS: csrr t2, mstatush
# name
csrrs t1, mstatush, zero
# uimm12
csrrs t2, 0x310, zero

#########################
# Machine Configuration
#########################

# menvcfgh
# name
# CHECK-INST: csrrs t1, menvcfgh, zero
# CHECK-ENC: encoding: [0x73,0x23,0xa0,0x31]
# CHECK-INST-ALIAS: csrr t1, menvcfgh
# uimm12
# CHECK-INST: csrrs t2, menvcfgh, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xa0,0x31]
# CHECK-INST-ALIAS: csrr t2, menvcfgh
# name
csrrs t1, menvcfgh, zero
# uimm12
csrrs t2, 0x31A, zero

# mseccfgh
# name
# CHECK-INST: csrrs t1, mseccfgh, zero
# CHECK-ENC: encoding: [0x73,0x23,0x70,0x75]
# CHECK-INST-ALIAS: csrr t1, mseccfgh
# uimm12
# CHECK-INST: csrrs t2, mseccfgh, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x70,0x75]
# CHECK-INST-ALIAS: csrr t2, mseccfgh
# name
csrrs t1, mseccfgh, zero
# uimm12
csrrs t2, 0x757, zero

######################################
# Machine Protection and Translation
######################################

# pmpcfg1
# name
# CHECK-INST: csrrs t1, pmpcfg1, zero
# CHECK-ENC: encoding: [0x73,0x23,0x10,0x3a]
# CHECK-INST-ALIAS: csrr t1, pmpcfg1
# uimm12
# CHECK-INST: csrrs t2, pmpcfg1, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x10,0x3a]
# CHECK-INST-ALIAS: csrr t2, pmpcfg1
# name
csrrs t1, pmpcfg1, zero
# uimm12
csrrs t2, 0x3A1, zero

# pmpcfg3
# name
# CHECK-INST: csrrs t1, pmpcfg3, zero
# CHECK-ENC: encoding: [0x73,0x23,0x30,0x3a]
# CHECK-INST-ALIAS: csrr t1, pmpcfg3
# uimm12
# CHECK-INST: csrrs t2, pmpcfg3, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x30,0x3a]
# CHECK-INST-ALIAS: csrr t2, pmpcfg3
# name
csrrs t1, pmpcfg3, zero
# uimm12
csrrs t2, 0x3A3, zero

# pmpcfg5
# name
# CHECK-INST: csrrs t1, pmpcfg5, zero
# CHECK-ENC: encoding: [0x73,0x23,0x50,0x3a]
# CHECK-INST-ALIAS: csrr t1, pmpcfg5
# uimm12
# CHECK-INST: csrrs t2, pmpcfg5, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x50,0x3a]
# CHECK-INST-ALIAS: csrr t2, pmpcfg5
# name
csrrs t1, pmpcfg5, zero
# uimm12
csrrs t2, 0x3A5, zero

# pmpcfg7
# name
# CHECK-INST: csrrs t1, pmpcfg7, zero
# CHECK-ENC: encoding: [0x73,0x23,0x70,0x3a]
# CHECK-INST-ALIAS: csrr t1, pmpcfg7
# uimm12
# CHECK-INST: csrrs t2, pmpcfg7, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x70,0x3a]
# CHECK-INST-ALIAS: csrr t2, pmpcfg7
# name
csrrs t1, pmpcfg7, zero
# uimm12
csrrs t2, 0x3A7, zero

# pmpcfg9
# name
# CHECK-INST: csrrs t1, pmpcfg9, zero
# CHECK-ENC: encoding: [0x73,0x23,0x90,0x3a]
# CHECK-INST-ALIAS: csrr t1, pmpcfg9
# uimm12
# CHECK-INST: csrrs t2, pmpcfg9, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x90,0x3a]
# CHECK-INST-ALIAS: csrr t2, pmpcfg9
# name
csrrs t1, pmpcfg9, zero
# uimm12
csrrs t2, 0x3A9, zero

# pmpcfg11
# name
# CHECK-INST: csrrs t1, pmpcfg11, zero
# CHECK-ENC: encoding: [0x73,0x23,0xb0,0x3a]
# CHECK-INST-ALIAS: csrr t1, pmpcfg11
# uimm12
# CHECK-INST: csrrs t2, pmpcfg11, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xb0,0x3a]
# CHECK-INST-ALIAS: csrr t2, pmpcfg11
# name
csrrs t1, pmpcfg11, zero
# uimm12
csrrs t2, 0x3AB, zero

# pmpcfg13
# name
# CHECK-INST: csrrs t1, pmpcfg13, zero
# CHECK-ENC: encoding: [0x73,0x23,0xd0,0x3a]
# CHECK-INST-ALIAS: csrr t1, pmpcfg13
# uimm12
# CHECK-INST: csrrs t2, pmpcfg13, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xd0,0x3a]
# CHECK-INST-ALIAS: csrr t2, pmpcfg13
# name
csrrs t1, pmpcfg13, zero
# uimm12
csrrs t2, 0x3AD, zero

# pmpcfg15
# name
# CHECK-INST: csrrs t1, pmpcfg15, zero
# CHECK-ENC: encoding: [0x73,0x23,0xf0,0x3a]
# CHECK-INST-ALIAS: csrr t1, pmpcfg15
# uimm12
# CHECK-INST: csrrs t2, pmpcfg15, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xf0,0x3a]
# CHECK-INST-ALIAS: csrr t2, pmpcfg15
# name
csrrs t1, pmpcfg15, zero
# uimm12
csrrs t2, 0x3AF, zero

######################################
# Machine Counter and Timers
######################################
# mcycleh
# name
# CHECK-INST: csrrs t1, mcycleh, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0xb8]
# CHECK-INST-ALIAS: csrr t1, mcycleh
# uimm12
# CHECK-INST: csrrs t2, mcycleh, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0xb8]
# CHECK-INST-ALIAS: csrr t2, mcycleh
csrrs t1, mcycleh, zero
# uimm12
csrrs t2, 0xB80, zero

# minstreth
# name
# CHECK-INST: csrrs t1, minstreth, zero
# CHECK-ENC: encoding: [0x73,0x23,0x20,0xb8]
# CHECK-INST-ALIAS: csrr t1, minstreth
# uimm12
# CHECK-INST: csrrs t2, minstreth, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x20,0xb8]
# CHECK-INST-ALIAS: csrr t2, minstreth
# name
csrrs t1, minstreth, zero
# uimm12
csrrs t2, 0xB82, zero

# mhpmcounter3h
# name
# CHECK-INST: csrrs t1, mhpmcounter3h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x30,0xb8]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter3h
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter3h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x30,0xb8]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter3h
# name
csrrs t1, mhpmcounter3h, zero
# uimm12
csrrs t2, 0xB83, zero

# mhpmcounter4h
# name
# CHECK-INST: csrrs t1, mhpmcounter4h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x40,0xb8]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter4h
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter4h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x40,0xb8]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter4h
# name
csrrs t1, mhpmcounter4h, zero
# uimm12
csrrs t2, 0xB84, zero

# mhpmcounter5h
# name
# CHECK-INST: csrrs t1, mhpmcounter5h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x50,0xb8]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter5h
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter5h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x50,0xb8]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter5h
# name
csrrs t1, mhpmcounter5h, zero
# uimm12
csrrs t2, 0xB85, zero

# mhpmcounter6h
# name
# CHECK-INST: csrrs t1, mhpmcounter6h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x60,0xb8]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter6h
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter6h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x60,0xb8]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter6h
# name
csrrs t1, mhpmcounter6h, zero
# uimm12
csrrs t2, 0xB86, zero

# mhpmcounter7h
# name
# CHECK-INST: csrrs t1, mhpmcounter7h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x70,0xb8]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter7h
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter7h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x70,0xb8]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter7h
# name
csrrs t1, mhpmcounter7h, zero
# uimm12
csrrs t2, 0xB87, zero

# mhpmcounter8h
# name
# CHECK-INST: csrrs t1, mhpmcounter8h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x80,0xb8]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter8h
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter8h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x80,0xb8]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter8h
# name
csrrs t1, mhpmcounter8h, zero
# uimm12
csrrs t2, 0xB88, zero

# mhpmcounter9h
# name
# CHECK-INST: csrrs t1, mhpmcounter9h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x90,0xb8]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter9h
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter9h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x90,0xb8]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter9h
# name
csrrs t1, mhpmcounter9h, zero
# uimm12
csrrs t2, 0xB89, zero

# mhpmcounter10h
# name
# CHECK-INST: csrrs t1, mhpmcounter10h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xa0,0xb8]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter10h
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter10h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xa0,0xb8]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter10h
# name
csrrs t1, mhpmcounter10h, zero
# uimm12
csrrs t2, 0xB8A, zero

# mhpmcounter11h
# name
# CHECK-INST: csrrs t1, mhpmcounter11h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xb0,0xb8]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter11h
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter11h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xb0,0xb8]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter11h
# name
csrrs t1, mhpmcounter11h, zero
# uimm12
csrrs t2, 0xB8B, zero

# mhpmcounter12h
# name
# CHECK-INST: csrrs t1, mhpmcounter12h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xc0,0xb8]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter12h
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter12h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xc0,0xb8]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter12h
# name
csrrs t1, mhpmcounter12h, zero
# uimm12
csrrs t2, 0xB8C, zero

# mhpmcounter13h
# name
# CHECK-INST: csrrs t1, mhpmcounter13h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xd0,0xb8]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter13h
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter13h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xd0,0xb8]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter13h
# name
csrrs t1, mhpmcounter13h, zero
# uimm12
csrrs t2, 0xB8D, zero

# mhpmcounter14h
# name
# CHECK-INST: csrrs t1, mhpmcounter14h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xe0,0xb8]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter14h
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter14h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xe0,0xb8]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter14h
# name
csrrs t1, mhpmcounter14h, zero
# uimm12
csrrs t2, 0xB8E, zero

# mhpmcounter15h
# name
# CHECK-INST: csrrs t1, mhpmcounter15h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xf0,0xb8]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter15h
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter15h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xf0,0xb8]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter15h
# name
csrrs t1, mhpmcounter15h, zero
# uimm12
csrrs t2, 0xB8F, zero

# mhpmcounter16h
# name
# CHECK-INST: csrrs t1, mhpmcounter16h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x00,0xb9]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter16h
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter16h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x00,0xb9]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter16h
# name
csrrs t1, mhpmcounter16h, zero
# uimm12
csrrs t2, 0xB90, zero

# mhpmcounter17h
# name
# CHECK-INST: csrrs t1, mhpmcounter17h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x10,0xb9]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter17h
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter17h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x10,0xb9]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter17h
# name
csrrs t1, mhpmcounter17h, zero
# uimm12
csrrs t2, 0xB91, zero

# mhpmcounter18h
# name
# CHECK-INST: csrrs t1, mhpmcounter18h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x20,0xb9]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter18h
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter18h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x20,0xb9]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter18h
# name
csrrs t1, mhpmcounter18h, zero
# uimm12
csrrs t2, 0xB92, zero

# mhpmcounter19h
# name
# CHECK-INST: csrrs t1, mhpmcounter19h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x30,0xb9]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter19h
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter19h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x30,0xb9]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter19h
# name
csrrs t1, mhpmcounter19h, zero
# uimm12
csrrs t2, 0xB93, zero

# mhpmcounter20h
# name
# CHECK-INST: csrrs t1, mhpmcounter20h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x40,0xb9]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter20h
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter20h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x40,0xb9]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter20h
# name
csrrs t1, mhpmcounter20h, zero
# uimm12
csrrs t2, 0xB94, zero

# mhpmcounter21h
# name
# CHECK-INST: csrrs t1, mhpmcounter21h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x50,0xb9]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter21h
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter21h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x50,0xb9]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter21h
# name
csrrs t1, mhpmcounter21h, zero
# uimm12
csrrs t2, 0xB95, zero

# mhpmcounter22h
# name
# CHECK-INST: csrrs t1, mhpmcounter22h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x60,0xb9]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter22h
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter22h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x60,0xb9]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter22h
# name
csrrs t1, mhpmcounter22h, zero
# uimm12
csrrs t2, 0xB96, zero

# mhpmcounter23h
# name
# CHECK-INST: csrrs t1, mhpmcounter23h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x70,0xb9]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter23h
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter23h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x70,0xb9]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter23h
# name
csrrs t1, mhpmcounter23h, zero
# uimm12
csrrs t2, 0xB97, zero

# mhpmcounter24h
# name
# CHECK-INST: csrrs t1, mhpmcounter24h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x80,0xb9]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter24h
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter24h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x80,0xb9]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter24h
# name
csrrs t1, mhpmcounter24h, zero
# uimm12
csrrs t2, 0xB98, zero

# mhpmcounter25h
# name
# CHECK-INST: csrrs t1, mhpmcounter25h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x90,0xb9]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter25h
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter25h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x90,0xb9]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter25h
# name
csrrs t1, mhpmcounter25h, zero
# uimm12
csrrs t2, 0xB99, zero

# mhpmcounter26h
# name
# CHECK-INST: csrrs t1, mhpmcounter26h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xa0,0xb9]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter26h
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter26h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xa0,0xb9]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter26h
# name
csrrs t1, mhpmcounter26h, zero
# uimm12
csrrs t2, 0xB9A, zero

# mhpmcounter27h
# name
# CHECK-INST: csrrs t1, mhpmcounter27h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xb0,0xb9]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter27h
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter27h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xb0,0xb9]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter27h
# name
csrrs t1, mhpmcounter27h, zero
# uimm12
csrrs t2, 0xB9B, zero

# mhpmcounter28h
# name
# CHECK-INST: csrrs t1, mhpmcounter28h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xc0,0xb9]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter28h
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter28h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xc0,0xb9]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter28h
# name
csrrs t1, mhpmcounter28h, zero
# uimm12
csrrs t2, 0xB9C, zero

# mhpmcounter29h
# name
# CHECK-INST: csrrs t1, mhpmcounter29h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xd0,0xb9]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter29h
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter29h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xd0,0xb9]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter29h
# name
csrrs t1, mhpmcounter29h, zero
# uimm12
csrrs t2, 0xB9D, zero

# mhpmcounter30h
# name
# CHECK-INST: csrrs t1, mhpmcounter30h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xe0,0xb9]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter30h
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter30h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xe0,0xb9]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter30h
# name
csrrs t1, mhpmcounter30h, zero
# uimm12
csrrs t2, 0xB9E, zero

# mhpmcounter31h
# name
# CHECK-INST: csrrs t1, mhpmcounter31h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xf0,0xb9]
# CHECK-INST-ALIAS: csrr t1, mhpmcounter31h
# uimm12
# CHECK-INST: csrrs t2, mhpmcounter31h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xf0,0xb9]
# CHECK-INST-ALIAS: csrr t2, mhpmcounter31h
# name
csrrs t1, mhpmcounter31h, zero
# uimm12
csrrs t2, 0xB9F, zero

######################################
# Machine Counter Setup
######################################

# mhpmevent3h
# name
# CHECK-INST: csrrs t1, mhpmevent3h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x30,0x72]
# CHECK-INST-ALIAS: csrr t1, mhpmevent3h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent3h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x30,0x72]
# CHECK-INST-ALIAS: csrr t2, mhpmevent3h
# name
csrrs t1, mhpmevent3h, zero
# uimm12
csrrs t2, 0x723, zero

# mhpmevent4h
# name
# CHECK-INST: csrrs t1, mhpmevent4h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x40,0x72]
# CHECK-INST-ALIAS: csrr t1, mhpmevent4h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent4h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x40,0x72]
# CHECK-INST-ALIAS: csrr t2, mhpmevent4h
# name
csrrs t1, mhpmevent4h, zero
# uimm12
csrrs t2, 0x724, zero

# mhpmevent5h
# name
# CHECK-INST: csrrs t1, mhpmevent5h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x50,0x72]
# CHECK-INST-ALIAS: csrr t1, mhpmevent5h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent5h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x50,0x72]
# CHECK-INST-ALIAS: csrr t2, mhpmevent5h
# name
csrrs t1, mhpmevent5h, zero
# uimm12
csrrs t2, 0x725, zero

# mhpmevent6h
# name
# CHECK-INST: csrrs t1, mhpmevent6h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x60,0x72]
# CHECK-INST-ALIAS: csrr t1, mhpmevent6h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent6h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x60,0x72]
# CHECK-INST-ALIAS: csrr t2, mhpmevent6h
# name
csrrs t1, mhpmevent6h, zero
# uimm12
csrrs t2, 0x726, zero

# mhpmevent7h
# name
# CHECK-INST: csrrs t1, mhpmevent7h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x70,0x72]
# CHECK-INST-ALIAS: csrr t1, mhpmevent7h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent7h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x70,0x72]
# CHECK-INST-ALIAS: csrr t2, mhpmevent7h
# name
csrrs t1, mhpmevent7h, zero
# uimm12
csrrs t2, 0x727, zero

# mhpmevent8h
# name
# CHECK-INST: csrrs t1, mhpmevent8h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x80,0x72]
# CHECK-INST-ALIAS: csrr t1, mhpmevent8h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent8h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x80,0x72]
# CHECK-INST-ALIAS: csrr t2, mhpmevent8h
# name
csrrs t1, mhpmevent8h, zero
# uimm12
csrrs t2, 0x728, zero

# mhpmevent9h
# name
# CHECK-INST: csrrs t1, mhpmevent9h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x90,0x72]
# CHECK-INST-ALIAS: csrr t1, mhpmevent9h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent9h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x90,0x72]
# CHECK-INST-ALIAS: csrr t2, mhpmevent9h
# name
csrrs t1, mhpmevent9h, zero
# uimm12
csrrs t2, 0x729, zero

# mhpmevent10h
# name
# CHECK-INST: csrrs t1, mhpmevent10h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xa0,0x72]
# CHECK-INST-ALIAS: csrr t1, mhpmevent10h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent10h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xa0,0x72]
# CHECK-INST-ALIAS: csrr t2, mhpmevent10h
# name
csrrs t1, mhpmevent10h, zero
# uimm12
csrrs t2, 0x72a, zero

# mhpmevent11h
# name
# CHECK-INST: csrrs t1, mhpmevent11h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xb0,0x72]
# CHECK-INST-ALIAS: csrr t1, mhpmevent11h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent11h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xb0,0x72]
# CHECK-INST-ALIAS: csrr t2, mhpmevent11h
# name
csrrs t1, mhpmevent11h, zero
# uimm12
csrrs t2, 0x72B, zero

# mhpmevent12h
# name
# CHECK-INST: csrrs t1, mhpmevent12h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xc0,0x72]
# CHECK-INST-ALIAS: csrr t1, mhpmevent12h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent12h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xc0,0x72]
# CHECK-INST-ALIAS: csrr t2, mhpmevent12h
# name
csrrs t1, mhpmevent12h, zero
# uimm12
csrrs t2, 0x72C, zero

# mhpmevent13h
# name
# CHECK-INST: csrrs t1, mhpmevent13h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xd0,0x72]
# CHECK-INST-ALIAS: csrr t1, mhpmevent13h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent13h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xd0,0x72]
# CHECK-INST-ALIAS: csrr t2, mhpmevent13h
# name
csrrs t1, mhpmevent13h, zero
# uimm12
csrrs t2, 0x72D, zero

# mhpmevent14h
# name
# CHECK-INST: csrrs t1, mhpmevent14h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xe0,0x72]
# CHECK-INST-ALIAS: csrr t1, mhpmevent14h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent14h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xe0,0x72]
# CHECK-INST-ALIAS: csrr t2, mhpmevent14h
# name
csrrs t1, mhpmevent14h, zero
# uimm12
csrrs t2, 0x72E, zero

# mhpmevent15h
# name
# CHECK-INST: csrrs t1, mhpmevent15h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xf0,0x72]
# CHECK-INST-ALIAS: csrr t1, mhpmevent15h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent15h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xf0,0x72]
# CHECK-INST-ALIAS: csrr t2, mhpmevent15h
# name
csrrs t1, mhpmevent15h, zero
# uimm12
csrrs t2, 0x72F, zero

# mhpmevent16h
# name
# CHECK-INST: csrrs t1, mhpmevent16h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x00,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent16h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent16h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x00,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent16h
# name
csrrs t1, mhpmevent16h, zero
# uimm12
csrrs t2, 0x730, zero

# mhpmevent17h
# name
# CHECK-INST: csrrs t1, mhpmevent17h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x10,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent17h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent17h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x10,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent17h
# name
csrrs t1, mhpmevent17h, zero
# uimm12
csrrs t2, 0x731, zero

# mhpmevent18h
# name
# CHECK-INST: csrrs t1, mhpmevent18h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x20,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent18h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent18h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x20,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent18h
# name
csrrs t1, mhpmevent18h, zero
# uimm12
csrrs t2, 0x732, zero

# mhpmevent19h
# name
# CHECK-INST: csrrs t1, mhpmevent19h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x30,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent19h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent19h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x30,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent19h
# name
csrrs t1, mhpmevent19h, zero
# uimm12
csrrs t2, 0x733, zero

# mhpmevent20h
# name
# CHECK-INST: csrrs t1, mhpmevent20h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x40,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent20h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent20h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x40,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent20h
# name
csrrs t1, mhpmevent20h, zero
# uimm12
csrrs t2, 0x734, zero

# mhpmevent21h
# name
# CHECK-INST: csrrs t1, mhpmevent21h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x50,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent21h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent21h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x50,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent21h
# name
csrrs t1, mhpmevent21h, zero
# uimm12
csrrs t2, 0x735, zero

# mhpmevent22h
# name
# CHECK-INST: csrrs t1, mhpmevent22h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x60,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent22h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent22h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x60,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent22h
# name
csrrs t1, mhpmevent22h, zero
# uimm12
csrrs t2, 0x736, zero

# mhpmevent23h
# name
# CHECK-INST: csrrs t1, mhpmevent23h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x70,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent23h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent23h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x70,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent23h
# name
csrrs t1, mhpmevent23h, zero
# uimm12
csrrs t2, 0x737, zero

# mhpmevent24h
# name
# CHECK-INST: csrrs t1, mhpmevent24h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x80,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent24h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent24h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x80,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent24h
# name
csrrs t1, mhpmevent24h, zero
# uimm12
csrrs t2, 0x738, zero

# mhpmevent25h
# name
# CHECK-INST: csrrs t1, mhpmevent25h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x90,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent25h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent25h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x90,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent25h
# name
csrrs t1, mhpmevent25h, zero
# uimm12
csrrs t2, 0x739, zero

# mhpmevent26h
# name
# CHECK-INST: csrrs t1, mhpmevent26h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xa0,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent26h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent26h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xa0,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent26h
# name
csrrs t1, mhpmevent26h, zero
# uimm12
csrrs t2, 0x73A, zero

# mhpmevent27h
# name
# CHECK-INST: csrrs t1, mhpmevent27h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xb0,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent27h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent27h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xb0,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent27h
# name
csrrs t1, mhpmevent27h, zero
# uimm12
csrrs t2, 0x73B, zero

# mhpmevent28h
# name
# CHECK-INST: csrrs t1, mhpmevent28h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xc0,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent28h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent28h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xc0,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent28h
# name
csrrs t1, mhpmevent28h, zero
# uimm12
csrrs t2, 0x73C, zero

# mhpmevent29h
# name
# CHECK-INST: csrrs t1, mhpmevent29h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xd0,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent29h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent29h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xd0,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent29h
# name
csrrs t1, mhpmevent29h, zero
# uimm12
csrrs t2, 0x73D, zero

# mhpmevent30h
# name
# CHECK-INST: csrrs t1, mhpmevent30h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xe0,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent30h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent30h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xe0,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent30h
# name
csrrs t1, mhpmevent30h, zero
# uimm12
csrrs t2, 0x73E, zero

# mhpmevent31h
# name
# CHECK-INST: csrrs t1, mhpmevent31h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xf0,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent31h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent31h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xf0,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent31h
# name
csrrs t1, mhpmevent31h, zero
# uimm12
csrrs t2, 0x73F, zero

#########################################
# State Enable Extension (Smstateen)
#########################################

# mstateen0h
# name
# CHECK-INST: csrrs t1, mstateen0h, zero
# CHECK-ENC: encoding: [0x73,0x23,0xc0,0x31]
# CHECK-INST-ALIAS: csrr t1, mstateen0h
# uimm12
# CHECK-INST: csrrs t2, mstateen0h, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xc0,0x31]
# CHECK-INST-ALIAS: csrr t2, mstateen0h
# name
csrrs t1, mstateen0h, zero
# uimm12
csrrs t2, 0x31C, zero

# mstateen1h
# name
# CHECK-INST: csrrs t1, mstateen1h, zero
# CHECK-ENC: encoding: [0x73,0x23,0xd0,0x31]
# CHECK-INST-ALIAS: csrr t1, mstateen1h
# uimm12
# CHECK-INST: csrrs t2, mstateen1h, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xd0,0x31]
# CHECK-INST-ALIAS: csrr t2, mstateen1h
# name
csrrs t1, mstateen1h, zero
# uimm12
csrrs t2, 0x31D, zero

# mstateen2h
# name
# CHECK-INST: csrrs t1, mstateen2h, zero
# CHECK-ENC: encoding: [0x73,0x23,0xe0,0x31]
# CHECK-INST-ALIAS: csrr t1, mstateen2h
# uimm12
# CHECK-INST: csrrs t2, mstateen2h, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xe0,0x31]
# CHECK-INST-ALIAS: csrr t2, mstateen2h
# name
csrrs t1, mstateen2h, zero
# uimm12
csrrs t2, 0x31E, zero

# mstateen3h
# name
# CHECK-INST: csrrs t1, mstateen3h, zero
# CHECK-ENC: encoding: [0x73,0x23,0xf0,0x31]
# CHECK-INST-ALIAS: csrr t1, mstateen3h
# uimm12
# CHECK-INST: csrrs t2, mstateen3h, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xf0,0x31]
# CHECK-INST-ALIAS: csrr t2, mstateen3h
# name
csrrs t1, mstateen3h, zero
# uimm12
csrrs t2, 0x31F, zero

#########################################
# Advanced Interrupt Architecture (Smaia and Ssaia)
#########################################

# midelegh
# name
# CHECK-INST: csrrs t1, midelegh, zero
# CHECK-ENC: encoding: [0x73,0x23,0x30,0x31]
# CHECK-INST-ALIAS: csrr t1, midelegh
# uimm12
# CHECK-INST: csrrs t2, midelegh, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x30,0x31]
# CHECK-INST-ALIAS: csrr t2, midelegh
# name
csrrs t1, midelegh, zero
# uimm12
csrrs t2, 0x313, zero

# mieh
# name
# CHECK-INST: csrrs t1, mieh, zero
# CHECK-ENC: encoding: [0x73,0x23,0x40,0x31]
# CHECK-INST-ALIAS: csrr t1, mieh
# uimm12
# CHECK-INST: csrrs t2, mieh, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x40,0x31]
# CHECK-INST-ALIAS: csrr t2, mieh
# name
csrrs t1, mieh, zero
# uimm12
csrrs t2, 0x314, zero

# mvienh
# name
# CHECK-INST: csrrs t1, mvienh, zero
# CHECK-ENC: encoding: [0x73,0x23,0x80,0x31]
# CHECK-INST-ALIAS: csrr t1, mvienh
# uimm12
# CHECK-INST: csrrs t2, mvienh, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x80,0x31]
# CHECK-INST-ALIAS: csrr t2, mvienh
# name
csrrs t1, mvienh, zero
# uimm12
csrrs t2, 0x318, zero

# mviph
# name
# CHECK-INST: csrrs t1, mviph, zero
# CHECK-ENC: encoding: [0x73,0x23,0x90,0x31]
# CHECK-INST-ALIAS: csrr t1, mviph
# uimm12
# CHECK-INST: csrrs t2, mviph, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x90,0x31]
# CHECK-INST-ALIAS: csrr t2, mviph
# name
csrrs t1, mviph, zero
# uimm12
csrrs t2, 0x319, zero

# miph
# name
# CHECK-INST: csrrs t1, miph, zero
# CHECK-ENC: encoding: [0x73,0x23,0x40,0x35]
# CHECK-INST-ALIAS: csrr t1, miph
# uimm12
# CHECK-INST: csrrs t2, miph, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x40,0x35]
# CHECK-INST-ALIAS: csrr t2, miph
# name
csrrs t1, miph, zero
# uimm12
csrrs t2, 0x354, zero
