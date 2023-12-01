# RUN: llvm-mc %s -triple=riscv64 -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s

# These user mode CSR register names are RV32 only, but RV64
# can encode and disassemble these registers if given their value.

##################################
# User Counter and Timers
##################################

# cycleh
# uimm12
# CHECK-INST: csrrs t2, 3200, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x00,0xc8]
# CHECK-INST-ALIAS: csrr t2, 0xc80
csrrs t2, 0xC80, zero

# timeh
# uimm12
# CHECK-INST: csrrs t2, 3201, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x10,0xc8]
# CHECK-INST-ALIAS: csrr t2, 0xc81
csrrs t2, 0xC81, zero

# instreth
# uimm12
# CHECK-INST: csrrs t2, 3202, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x20,0xc8]
# CHECK-INST-ALIAS: csrr t2, 0xc82
csrrs t2, 0xC82, zero

# hpmcounter3h
# uimm12
# CHECK-INST: csrrs t2, 3203, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x30,0xc8]
# CHECK-INST-ALIAS: csrr t2, 0xc83
csrrs t2, 0xC83, zero

# hpmcounter4h
# uimm12
# CHECK-INST: csrrs t2, 3204, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x40,0xc8]
# CHECK-INST-ALIAS: csrr t2, 0xc84
csrrs t2, 0xC84, zero

# hpmcounter5h
# uimm12
# CHECK-INST: csrrs t2, 3205, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x50,0xc8]
# CHECK-INST-ALIAS: csrr t2, 0xc85
csrrs t2, 0xC85, zero

# hpmcounter6h
# uimm12
# CHECK-INST: csrrs t2, 3206, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x60,0xc8]
# CHECK-INST-ALIAS: csrr t2, 0xc86
csrrs t2, 0xC86, zero

# hpmcounter7h
# uimm12
# CHECK-INST: csrrs t2, 3207, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x70,0xc8]
# CHECK-INST-ALIAS: csrr t2, 0xc87
csrrs t2, 0xC87, zero

# hpmcounter8h
# uimm12
# CHECK-INST: csrrs t2, 3208, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x80,0xc8]
# CHECK-INST-ALIAS: csrr t2, 0xc88
csrrs t2, 0xC88, zero

# hpmcounter9h
# uimm12
# CHECK-INST: csrrs t2, 3209, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x90,0xc8]
# CHECK-INST-ALIAS: csrr t2, 0xc89
csrrs t2, 0xC89, zero

# hpmcounter10h
# uimm12
# CHECK-INST: csrrs t2, 3210, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xa0,0xc8]
# CHECK-INST-ALIAS: csrr t2, 0xc8a
csrrs t2, 0xC8A, zero

# hpmcounter11h
# uimm12
# CHECK-INST: csrrs t2, 3211, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xb0,0xc8]
# CHECK-INST-ALIAS: csrr t2, 0xc8b
csrrs t2, 0xC8B, zero

# hpmcounter12h
# uimm12
# CHECK-INST: csrrs t2, 3212, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xc0,0xc8]
# CHECK-INST-ALIAS: csrr t2, 0xc8c
csrrs t2, 0xC8C, zero

# hpmcounter13h
# uimm12
# CHECK-INST: csrrs t2, 3213, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xd0,0xc8]
# CHECK-INST-ALIAS: csrr t2, 0xc8d
csrrs t2, 0xC8D, zero

# hpmcounter14h
# uimm12
# CHECK-INST: csrrs t2, 3214, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xe0,0xc8]
# CHECK-INST-ALIAS: csrr t2, 0xc8e
csrrs t2, 0xC8E, zero

# hpmcounter15h
# uimm12
# CHECK-INST: csrrs t2, 3215, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xf0,0xc8]
# CHECK-INST-ALIAS: csrr t2, 0xc8f
csrrs t2, 0xC8F, zero

# hpmcounter16h
# uimm12
# CHECK-INST: csrrs t2, 3216, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x00,0xc9]
# CHECK-INST-ALIAS: csrr t2, 0xc90
csrrs t2, 0xC90, zero

# hpmcounter17h
# uimm12
# CHECK-INST: csrrs t2, 3217, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x10,0xc9]
# CHECK-INST-ALIAS: csrr t2, 0xc91
csrrs t2, 0xC91, zero

# hpmcounter18h
# uimm12
# CHECK-INST: csrrs t2, 3218, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x20,0xc9]
# CHECK-INST-ALIAS: csrr t2, 0xc92
csrrs t2, 0xC92, zero

# hpmcounter19h
# uimm12
# CHECK-INST: csrrs t2, 3219, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x30,0xc9]
# CHECK-INST-ALIAS: csrr t2, 0xc93
csrrs t2, 0xC93, zero

# hpmcounter20h
# uimm12
# CHECK-INST: csrrs t2, 3220, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x40,0xc9]
# CHECK-INST-ALIAS: csrr t2, 0xc94
csrrs t2, 0xC94, zero

# hpmcounter21h
# uimm12
# CHECK-INST: csrrs t2, 3221, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x50,0xc9]
# CHECK-INST-ALIAS: csrr t2, 0xc95
csrrs t2, 0xC95, zero

# hpmcounter22h
# uimm12
# CHECK-INST: csrrs t2, 3222, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x60,0xc9]
# CHECK-INST-ALIAS: csrr t2, 0xc96
csrrs t2, 0xC96, zero

# hpmcounter23h
# uimm12
# CHECK-INST: csrrs t2, 3223, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x70,0xc9]
# CHECK-INST-ALIAS: csrr t2, 0xc97
csrrs t2, 0xC97, zero

# hpmcounter24h
# uimm12
# CHECK-INST: csrrs t2, 3224, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x80,0xc9]
# CHECK-INST-ALIAS: csrr t2, 0xc98
csrrs t2, 0xC98, zero

# hpmcounter25h
# uimm12
# CHECK-INST: csrrs t2, 3225, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x90,0xc9]
# CHECK-INST-ALIAS: csrr t2, 0xc99
csrrs t2, 0xC99, zero

# hpmcounter26h
# uimm12
# CHECK-INST: csrrs t2, 3226, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xa0,0xc9]
# CHECK-INST-ALIAS: csrr t2, 0xc9a
csrrs t2, 0xC9A, zero

# hpmcounter27h
# uimm12
# CHECK-INST: csrrs t2, 3227, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xb0,0xc9]
# CHECK-INST-ALIAS: csrr t2, 0xc9b
csrrs t2, 0xC9B, zero

# hpmcounter28h
# uimm12
# CHECK-INST: csrrs t2, 3228, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xc0,0xc9]
# CHECK-INST-ALIAS: csrr t2, 0xc9c
csrrs t2, 0xC9C, zero

# hpmcounter29h
# uimm12
# CHECK-INST: csrrs t2, 3229, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xd0,0xc9]
# CHECK-INST-ALIAS: csrr t2, 0xc9d
csrrs t2, 0xC9D, zero

# hpmcounter30h
# uimm12
# CHECK-INST: csrrs t2, 3230, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xe0,0xc9]
# CHECK-INST-ALIAS: csrr t2, 0xc9e
csrrs t2, 0xC9E, zero

# hpmcounter31h
# uimm12
# CHECK-INST: csrrs t2, 3231, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xf0,0xc9]
# CHECK-INST-ALIAS: csrr t2, 0xc9f
csrrs t2, 0xC9F, zero
