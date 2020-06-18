// RUN: llvm-mc -triple mips64-unknown-linux < %s -show-encoding \
// RUN:     | FileCheck -check-prefixes=ENCBE,FIXUP %s
// RUN: llvm-mc -triple mips64el-unknown-linux < %s -show-encoding \
// RUN:     | FileCheck -check-prefixes=ENCLE,FIXUP %s
// RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux < %s \
// RUN:     | llvm-readobj -r | FileCheck -check-prefix=RELOC %s
// RUN: llvm-mc -filetype=obj -triple mips64-unknown-linux < %s \
// RUN:     | llvm-readobj --sections --section-data \
// RUN:     | FileCheck -check-prefix=DATA %s

// Test that we produce the correct relocation.
// FIXME: move more relocation only tests here.

// Check prefixes:
// RELOC - Check the relocation in the object.
// FIXUP - Check the fixup on the instruction.
// ENCBE - Check the big-endian encoding on the instruction.
// ENCLE - Check the little-endian encoding on the instruction.
// ????? - Placeholder. Relocation is defined but the way of generating it is
//         unknown.
// FIXME - Placeholder. Generation method is known but doesn't work.

// DATA-LABEL: Name: .text
// DATA:       SectionData (

// DATA-NEXT:  0000: 3C030000 24620000 3C030000 24620000
        lui   $3, %hi(%neg(%gp_rel(foo)))     // RELOC: R_MIPS_GPREL16/R_MIPS_SUB/R_MIPS_HI16 foo
                                              // ENCBE: lui $3, %hi(%neg(%gp_rel(foo))) # encoding: [0x3c,0x03,A,A]
                                              // ENCLE: lui $3, %hi(%neg(%gp_rel(foo))) # encoding: [A,A,0x03,0x3c]
                                              // FIXUP: # fixup A - offset: 0, value: %hi(%neg(%gp_rel(foo))), kind: fixup_Mips_GPOFF_HI

        addiu $2, $3, %lo(%neg(%gp_rel(foo))) // RELOC: R_MIPS_GPREL16/R_MIPS_SUB/R_MIPS_LO16 foo
                                              // ENCBE: addiu $2, $3, %lo(%neg(%gp_rel(foo))) # encoding: [0x24,0x62,A,A]
                                              // ENCLE: addiu $2, $3, %lo(%neg(%gp_rel(foo))) # encoding: [A,A,0x62,0x24]
                                              // FIXUP: # fixup A - offset: 0, value: %lo(%neg(%gp_rel(foo))), kind: fixup_Mips_GPOFF_LO

        lui   $3, %hi(%neg(%gp_rel(bar)))     // RELOC: R_MIPS_GPREL16/R_MIPS_SUB/R_MIPS_HI16 .data
                                              // ENCBE: lui $3, %hi(%neg(%gp_rel(bar))) # encoding: [0x3c,0x03,A,A]
                                              // ENCLE: lui $3, %hi(%neg(%gp_rel(bar))) # encoding: [A,A,0x03,0x3c]
                                              // FIXUP: # fixup A - offset: 0, value: %hi(%neg(%gp_rel(bar))), kind: fixup_Mips_GPOFF_HI

        addiu $2, $3, %lo(%neg(%gp_rel(bar))) // RELOC: R_MIPS_GPREL16/R_MIPS_SUB/R_MIPS_LO16 .data
                                              // ENCBE: addiu $2, $3, %lo(%neg(%gp_rel(bar))) # encoding: [0x24,0x62,A,A]
                                              // ENCLE: addiu $2, $3, %lo(%neg(%gp_rel(bar))) # encoding: [A,A,0x62,0x24]
                                              // FIXUP: # fixup A - offset: 0, value: %lo(%neg(%gp_rel(bar))), kind: fixup_Mips_GPOFF_LO

// DATA-NEXT:  0010: 41A30000 30430000 41A30000 30430000
        .set micromips
        lui   $3, %hi(%neg(%gp_rel(foo)))     // RELOC: R_MICROMIPS_GPREL16/R_MICROMIPS_SUB/R_MICROMIPS_HI16 foo
                                              // ENCBE: lui $3, %hi(%neg(%gp_rel(foo))) # encoding: [0x41,0xa3,A,A]
                                              // ENCLE: lui $3, %hi(%neg(%gp_rel(foo))) # encoding: [0xa3'A',0x41'A',0x00,0x00]
                                              // FIXUP: # fixup A - offset: 0, value: %hi(%neg(%gp_rel(foo))), kind: fixup_MICROMIPS_GPOFF_HI

        addiu $2, $3, %lo(%neg(%gp_rel(foo))) // RELOC: R_MICROMIPS_GPREL16/R_MICROMIPS_SUB/R_MICROMIPS_LO16 foo
                                              // ENCBE: addiu $2, $3, %lo(%neg(%gp_rel(foo))) # encoding: [0x30,0x43,A,A]
                                              // ENCLE: addiu $2, $3, %lo(%neg(%gp_rel(foo))) # encoding: [0x43'A',0x30'A',0x00,0x00]
                                              // FIXUP: # fixup A - offset: 0, value: %lo(%neg(%gp_rel(foo))), kind: fixup_MICROMIPS_GPOFF_LO

        lui   $3, %hi(%neg(%gp_rel(bar)))     // RELOC: R_MICROMIPS_GPREL16/R_MICROMIPS_SUB/R_MICROMIPS_HI16 bar
                                              // ENCBE: lui $3, %hi(%neg(%gp_rel(bar))) # encoding: [0x41,0xa3,A,A]
                                              // ENCLE: lui $3, %hi(%neg(%gp_rel(bar))) # encoding: [0xa3'A',0x41'A',0x00,0x00]
                                              // FIXUP: # fixup A - offset: 0, value: %hi(%neg(%gp_rel(bar))), kind: fixup_MICROMIPS_GPOFF_HI

        addiu $2, $3, %lo(%neg(%gp_rel(bar))) // RELOC: R_MICROMIPS_GPREL16/R_MICROMIPS_SUB/R_MICROMIPS_LO16 bar
                                              // ENCBE: addiu $2, $3, %lo(%neg(%gp_rel(bar))) # encoding: [0x30,0x43,A,A]
                                              // ENCLE: addiu $2, $3, %lo(%neg(%gp_rel(bar))) # encoding: [0x43'A',0x30'A',0x00,0x00]
                                              // FIXUP: # fixup A - offset: 0, value: %lo(%neg(%gp_rel(bar))), kind: fixup_MICROMIPS_GPOFF_LO

        .data
        .word 0
bar:
        .word 1
        .option pic2
        .quad foo+0x1fffffff0-.               // RELOC: R_MIPS_PC32/R_MIPS_64/R_MIPS_NONE foo 0x1FFFFFFF0
// DATA-LABEL: Section {
