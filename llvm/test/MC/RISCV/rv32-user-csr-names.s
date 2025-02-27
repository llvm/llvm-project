# RUN: llvm-mc %s -triple=riscv32 -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s

##################################
# User Counter and Timers
##################################

# cycleh
# name
# CHECK-INST: csrrs t1, cycleh, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x00,0xc8]
# CHECK-INST-ALIAS: rdcycleh t1
# uimm12
# CHECK-INST: csrrs t2, cycleh, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x00,0xc8]
# CHECK-INST-ALIAS: rdcycleh t2
# name
csrrs t1, cycleh, zero
# uimm12
csrrs t2, 0xC80, zero

# timeh
# name
# CHECK-INST: csrrs t1, timeh, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x10,0xc8]
# CHECK-INST-ALIAS: rdtimeh t1
# uimm12
# CHECK-INST: csrrs t2, timeh, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x10,0xc8]
# CHECK-INST-ALIAS: rdtimeh t2
# name
csrrs t1, timeh, zero
# uimm12
csrrs t2, 0xC81, zero

# instreth
# name
# CHECK-INST: csrrs t1, instreth, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x20,0xc8]
# CHECK-INST-ALIAS: rdinstreth t1
# uimm12
# CHECK-INST: csrrs t2, instreth, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x20,0xc8]
# CHECK-INST-ALIAS: rdinstreth t2
# name
csrrs t1, instreth, zero
# uimm12
csrrs t2, 0xC82, zero

# hpmcounter3h
# name
# CHECK-INST: csrrs t1, hpmcounter3h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x30,0xc8]
# CHECK-INST-ALIAS: csrr t1, hpmcounter3h
# uimm12
# CHECK-INST: csrrs t2, hpmcounter3h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x30,0xc8]
# CHECK-INST-ALIAS: csrr t2, hpmcounter3h
# name
csrrs t1, hpmcounter3h, zero
# uimm12
csrrs t2, 0xC83, zero

# hpmcounter4h
# name
# CHECK-INST: csrrs t1, hpmcounter4h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x40,0xc8]
# CHECK-INST-ALIAS: csrr t1, hpmcounter4h
# uimm12
# CHECK-INST: csrrs t2, hpmcounter4h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x40,0xc8]
# CHECK-INST-ALIAS: csrr t2, hpmcounter4h
# name
csrrs t1, hpmcounter4h, zero
# uimm12
csrrs t2, 0xC84, zero

# hpmcounter5h
# name
# CHECK-INST: csrrs t1, hpmcounter5h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x50,0xc8]
# CHECK-INST-ALIAS: csrr t1, hpmcounter5h
# uimm12
# CHECK-INST: csrrs t2, hpmcounter5h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x50,0xc8]
# CHECK-INST-ALIAS: csrr t2, hpmcounter5h
# name
csrrs t1, hpmcounter5h, zero
# uimm12
csrrs t2, 0xC85, zero

# hpmcounter6h
# name
# CHECK-INST: csrrs t1, hpmcounter6h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x60,0xc8]
# CHECK-INST-ALIAS: csrr t1, hpmcounter6h
# uimm12
# CHECK-INST: csrrs t2, hpmcounter6h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x60,0xc8]
# CHECK-INST-ALIAS: csrr t2, hpmcounter6h
# name
csrrs t1, hpmcounter6h, zero
# uimm12
csrrs t2, 0xC86, zero

# hpmcounter7h
# name
# CHECK-INST: csrrs t1, hpmcounter7h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x70,0xc8]
# CHECK-INST-ALIAS: csrr t1, hpmcounter7h
# uimm12
# CHECK-INST: csrrs t2, hpmcounter7h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x70,0xc8]
# CHECK-INST-ALIAS: csrr t2, hpmcounter7h
# name
csrrs t1, hpmcounter7h, zero
# uimm12
csrrs t2, 0xC87, zero

# hpmcounter8h
# name
# CHECK-INST: csrrs t1, hpmcounter8h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x80,0xc8]
# CHECK-INST-ALIAS: csrr t1, hpmcounter8h
# uimm12
# CHECK-INST: csrrs t2, hpmcounter8h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x80,0xc8]
# CHECK-INST-ALIAS: csrr t2, hpmcounter8h
# name
csrrs t1, hpmcounter8h, zero
# uimm12
csrrs t2, 0xC88, zero

# hpmcounter9h
# name
# CHECK-INST: csrrs t1, hpmcounter9h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x90,0xc8]
# CHECK-INST-ALIAS: csrr t1, hpmcounter9h
# uimm12
# CHECK-INST: csrrs t2, hpmcounter9h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x90,0xc8]
# CHECK-INST-ALIAS: csrr t2, hpmcounter9h
# name
csrrs t1, hpmcounter9h, zero
# uimm12
csrrs t2, 0xC89, zero

# hpmcounter10h
# name
# CHECK-INST: csrrs t1, hpmcounter10h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xa0,0xc8]
# CHECK-INST-ALIAS: csrr t1, hpmcounter10h
# uimm12
# CHECK-INST: csrrs t2, hpmcounter10h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xa0,0xc8]
# CHECK-INST-ALIAS: csrr t2, hpmcounter10h
# name
csrrs t1, hpmcounter10h, zero
# uimm12
csrrs t2, 0xC8A, zero

# hpmcounter11h
# name
# CHECK-INST: csrrs t1, hpmcounter11h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xb0,0xc8]
# CHECK-INST-ALIAS: csrr t1, hpmcounter11h
# uimm12
# CHECK-INST: csrrs t2, hpmcounter11h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xb0,0xc8]
# CHECK-INST-ALIAS: csrr t2, hpmcounter11h
# name
csrrs t1, hpmcounter11h, zero
# uimm12
csrrs t2, 0xC8B, zero

# hpmcounter12h
# name
# CHECK-INST: csrrs t1, hpmcounter12h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xc0,0xc8]
# CHECK-INST-ALIAS: csrr t1, hpmcounter12h
# uimm12
# CHECK-INST: csrrs t2, hpmcounter12h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xc0,0xc8]
# CHECK-INST-ALIAS: csrr t2, hpmcounter12h
# name
csrrs t1, hpmcounter12h, zero
# uimm12
csrrs t2, 0xC8C, zero

# hpmcounter13h
# name
# CHECK-INST: csrrs t1, hpmcounter13h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xd0,0xc8]
# CHECK-INST-ALIAS: csrr t1, hpmcounter13h
# uimm12
# CHECK-INST: csrrs t2, hpmcounter13h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xd0,0xc8]
# CHECK-INST-ALIAS: csrr t2, hpmcounter13h
# name
csrrs t1, hpmcounter13h, zero
# uimm12
csrrs t2, 0xC8D, zero

# hpmcounter14h
# name
# CHECK-INST: csrrs t1, hpmcounter14h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xe0,0xc8]
# CHECK-INST-ALIAS: csrr t1, hpmcounter14h
# uimm12
# CHECK-INST: csrrs t2, hpmcounter14h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xe0,0xc8]
# CHECK-INST-ALIAS: csrr t2, hpmcounter14h
# name
csrrs t1, hpmcounter14h, zero
# uimm12
csrrs t2, 0xC8E, zero

# hpmcounter15h
# name
# CHECK-INST: csrrs t1, hpmcounter15h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xf0,0xc8]
# CHECK-INST-ALIAS: csrr t1, hpmcounter15h
# uimm12
# CHECK-INST: csrrs t2, hpmcounter15h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xf0,0xc8]
# CHECK-INST-ALIAS: csrr t2, hpmcounter15h
# name
csrrs t1, hpmcounter15h, zero
# uimm12
csrrs t2, 0xC8F, zero

# hpmcounter16h
# name
# CHECK-INST: csrrs t1, hpmcounter16h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x00,0xc9]
# CHECK-INST-ALIAS: csrr t1, hpmcounter16h
# uimm12
# CHECK-INST: csrrs t2, hpmcounter16h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x00,0xc9]
# CHECK-INST-ALIAS: csrr t2, hpmcounter16h
# name
csrrs t1, hpmcounter16h, zero
# uimm12
csrrs t2, 0xC90, zero

# hpmcounter17h
# name
# CHECK-INST: csrrs t1, hpmcounter17h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x10,0xc9]
# CHECK-INST-ALIAS: csrr t1, hpmcounter17h
# uimm12
# CHECK-INST: csrrs t2, hpmcounter17h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x10,0xc9]
# CHECK-INST-ALIAS: csrr t2, hpmcounter17h
# name
csrrs t1, hpmcounter17h, zero
# uimm12
csrrs t2, 0xC91, zero

# hpmcounter18h
# name
# CHECK-INST: csrrs t1, hpmcounter18h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x20,0xc9]
# CHECK-INST-ALIAS: csrr t1, hpmcounter18h
# uimm12
# CHECK-INST: csrrs t2, hpmcounter18h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x20,0xc9]
# CHECK-INST-ALIAS: csrr t2, hpmcounter18h
# name
csrrs t1, hpmcounter18h, zero
# uimm12
csrrs t2, 0xC92, zero

# hpmcounter19h
# name
# CHECK-INST: csrrs t1, hpmcounter19h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x30,0xc9]
# CHECK-INST-ALIAS: csrr t1, hpmcounter19h
# uimm12
# CHECK-INST: csrrs t2, hpmcounter19h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x30,0xc9]
# CHECK-INST-ALIAS: csrr t2, hpmcounter19h
# name
csrrs t1, hpmcounter19h, zero
# uimm12
csrrs t2, 0xC93, zero

# hpmcounter20h
# name
# CHECK-INST: csrrs t1, hpmcounter20h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x40,0xc9]
# CHECK-INST-ALIAS: csrr t1, hpmcounter20h
# uimm12
# CHECK-INST: csrrs t2, hpmcounter20h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x40,0xc9]
# CHECK-INST-ALIAS: csrr t2, hpmcounter20h
# name
csrrs t1, hpmcounter20h, zero
# uimm12
csrrs t2, 0xC94, zero

# hpmcounter21h
# name
# CHECK-INST: csrrs t1, hpmcounter21h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x50,0xc9]
# CHECK-INST-ALIAS: csrr t1, hpmcounter21h
# uimm12
# CHECK-INST: csrrs t2, hpmcounter21h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x50,0xc9]
# CHECK-INST-ALIAS: csrr t2, hpmcounter21h
# name
csrrs t1, hpmcounter21h, zero
# uimm12
csrrs t2, 0xC95, zero

# hpmcounter22h
# name
# CHECK-INST: csrrs t1, hpmcounter22h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x60,0xc9]
# CHECK-INST-ALIAS: csrr t1, hpmcounter22h
# uimm12
# CHECK-INST: csrrs t2, hpmcounter22h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x60,0xc9]
# CHECK-INST-ALIAS: csrr t2, hpmcounter22h
# name
csrrs t1, hpmcounter22h, zero
# uimm12
csrrs t2, 0xC96, zero

# hpmcounter23h
# name
# CHECK-INST: csrrs t1, hpmcounter23h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x70,0xc9]
# CHECK-INST-ALIAS: csrr t1, hpmcounter23h
# uimm12
# CHECK-INST: csrrs t2, hpmcounter23h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x70,0xc9]
# CHECK-INST-ALIAS: csrr t2, hpmcounter23h
# name
csrrs t1, hpmcounter23h, zero
# uimm12
csrrs t2, 0xC97, zero

# hpmcounter24h
# name
# CHECK-INST: csrrs t1, hpmcounter24h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x80,0xc9]
# CHECK-INST-ALIAS: csrr t1, hpmcounter24h
# uimm12
# CHECK-INST: csrrs t2, hpmcounter24h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x80,0xc9]
# CHECK-INST-ALIAS: csrr t2, hpmcounter24h
# name
csrrs t1, hpmcounter24h, zero
# uimm12
csrrs t2, 0xC98, zero

# hpmcounter25h
# name
# CHECK-INST: csrrs t1, hpmcounter25h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x90,0xc9]
# CHECK-INST-ALIAS: csrr t1, hpmcounter25h
# uimm12
# CHECK-INST: csrrs t2, hpmcounter25h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x90,0xc9]
# CHECK-INST-ALIAS: csrr t2, hpmcounter25h
# name
csrrs t1, hpmcounter25h, zero
# uimm12
csrrs t2, 0xC99, zero

# hpmcounter26h
# name
# CHECK-INST: csrrs t1, hpmcounter26h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xa0,0xc9]
# CHECK-INST-ALIAS: csrr t1, hpmcounter26h
# uimm12
# CHECK-INST: csrrs t2, hpmcounter26h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xa0,0xc9]
# CHECK-INST-ALIAS: csrr t2, hpmcounter26h
# name
csrrs t1, hpmcounter26h, zero
# uimm12
csrrs t2, 0xC9A, zero

# hpmcounter27h
# name
# CHECK-INST: csrrs t1, hpmcounter27h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xb0,0xc9]
# CHECK-INST-ALIAS: csrr t1, hpmcounter27h
# uimm12
# CHECK-INST: csrrs t2, hpmcounter27h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xb0,0xc9]
# CHECK-INST-ALIAS: csrr t2, hpmcounter27h
# name
csrrs t1, hpmcounter27h, zero
# uimm12
csrrs t2, 0xC9B, zero

# hpmcounter28h
# name
# CHECK-INST: csrrs t1, hpmcounter28h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xc0,0xc9]
# CHECK-INST-ALIAS: csrr t1, hpmcounter28h
# uimm12
# CHECK-INST: csrrs t2, hpmcounter28h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xc0,0xc9]
# CHECK-INST-ALIAS: csrr t2, hpmcounter28h
# name
csrrs t1, hpmcounter28h, zero
# uimm12
csrrs t2, 0xC9C, zero

# hpmcounter29h
# name
# CHECK-INST: csrrs t1, hpmcounter29h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xd0,0xc9]
# CHECK-INST-ALIAS: csrr t1, hpmcounter29h
# uimm12
# CHECK-INST: csrrs t2, hpmcounter29h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xd0,0xc9]
# CHECK-INST-ALIAS: csrr t2, hpmcounter29h
# name
csrrs t1, hpmcounter29h, zero
# uimm12
csrrs t2, 0xC9D, zero

# hpmcounter30h
# name
# CHECK-INST: csrrs t1, hpmcounter30h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xe0,0xc9]
# CHECK-INST-ALIAS: csrr t1, hpmcounter30h
# uimm12
# CHECK-INST: csrrs t2, hpmcounter30h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xe0,0xc9]
# CHECK-INST-ALIAS: csrr t2, hpmcounter30h
# name
csrrs t1, hpmcounter30h, zero
# uimm12
csrrs t2, 0xC9E, zero

# hpmcounter31h
# name
# CHECK-INST: csrrs t1, hpmcounter31h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xf0,0xc9]
# CHECK-INST-ALIAS: csrr t1, hpmcounter31h
# uimm12
# CHECK-INST: csrrs t2, hpmcounter31h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xf0,0xc9]
# CHECK-INST-ALIAS: csrr t2, hpmcounter31h
# name
csrrs t1, hpmcounter31h, zero
# uimm12
csrrs t2, 0xC9F, zero
