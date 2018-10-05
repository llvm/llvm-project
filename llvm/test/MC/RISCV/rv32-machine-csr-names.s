# RUN: llvm-mc %s -triple=riscv32 -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s

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

