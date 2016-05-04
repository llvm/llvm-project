// RUN: llvm-mc -triple mips-unknown-linux < %s -show-encoding \
// RUN:     | FileCheck -check-prefix=ENCBE -check-prefix=FIXUP %s
// RUN: llvm-mc -triple mipsel-unknown-linux < %s -show-encoding \
// RUN:     | FileCheck -check-prefix=ENCLE -check-prefix=FIXUP %s
// RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux < %s \
// RUN:     | llvm-readobj -r | FileCheck -check-prefix=RELOC %s

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

        .short foo                         // RELOC: R_MIPS_16 foo

        .long foo                          // RELOC: R_MIPS_32 foo

                                           // ?????: R_MIPS_REL32 foo

        jal foo                            // RELOC: R_MIPS_26 foo
                                           // ENCBE: jal foo # encoding: [0b000011AA,A,A,A]
                                           // ENCLE: jal foo # encoding: [A,A,A,0b000011AA]
                                           // FIXUP: # fixup A - offset: 0, value: foo, kind: fixup_Mips_26

        addiu $2, $3, %hi(foo)             // RELOC: R_MIPS_HI16 foo
                                           // ENCBE: addiu $2, $3, %hi(foo) # encoding: [0x24,0x62,A,A]
                                           // ENCLE: addiu $2, $3, %hi(foo) # encoding: [A,A,0x62,0x24]
                                           // FIXUP: # fixup A - offset: 0, value: %hi(foo), kind: fixup_Mips_HI16

        addiu $2, $3, %lo(foo)             // RELOC: R_MIPS_LO16 foo
                                           // ENCBE: addiu $2, $3, %lo(foo) # encoding: [0x24,0x62,A,A]
                                           // ENCLE: addiu $2, $3, %lo(foo) # encoding: [A,A,0x62,0x24]
                                           // FIXUP: # fixup A - offset: 0, value: %lo(foo), kind: fixup_Mips_LO16

        addiu $2, $3, %gp_rel(foo)         // RELOC: R_MIPS_GPREL16 foo
                                           // ENCBE: addiu $2, $3, %gp_rel(foo) # encoding: [0x24,0x62,A,A]
                                           // ENCLE: addiu $2, $3, %gp_rel(foo) # encoding: [A,A,0x62,0x24]
                                           // FIXUP: # fixup A - offset: 0, value: %gp_rel(foo), kind: fixup_Mips_GPREL

                                           // ?????: R_MIPS_LITERAL foo

        addiu $2, $3, %got(foo)            // RELOC: R_MIPS_GOT16 foo
                                           // ENCBE: addiu $2, $3, %got(foo) # encoding: [0x24,0x62,A,A]
                                           // ENCLE: addiu $2, $3, %got(foo) # encoding: [A,A,0x62,0x24]
                                           // FIXUP: # fixup A - offset: 0, value: %got(foo), kind: fixup_Mips_GOT

        .short foo-.                       // RELOC: R_MIPS_PC16 foo

        addiu $2, $3, %call16(foo)         // RELOC: R_MIPS_CALL16 foo
                                           // ENCBE: addiu $2, $3, %call16(foo) # encoding: [0x24,0x62,A,A]
                                           // ENCLE: addiu $2, $3, %call16(foo) # encoding: [A,A,0x62,0x24]
                                           // FIXUP: # fixup A - offset: 0, value: %call16(foo), kind: fixup_Mips_CALL16

        .quad foo                          // RELOC: R_MIPS_64 foo

                                           // ?????: R_MIPS_GPREL32 foo
                                           // ?????: R_MIPS_UNUSED1 foo
                                           // ?????: R_MIPS_UNUSED2 foo
                                           // ?????: R_MIPS_UNUSED3 foo
                                           // ?????: R_MIPS_SHIFT5 foo
                                           // ?????: R_MIPS_SHIFT6 foo

        addiu $2, $3, %got_disp(foo)       // RELOC: R_MIPS_GOT_DISP foo
                                           // ENCBE: addiu $2, $3, %got_disp(foo) # encoding: [0x24,0x62,A,A]
                                           // ENCLE: addiu $2, $3, %got_disp(foo) # encoding: [A,A,0x62,0x24]
                                           // FIXUP: # fixup A - offset: 0, value: %got_disp(foo), kind: fixup_Mips_GOT_DISP

        addiu $2, $3, %got_page(foo)       // RELOC: R_MIPS_GOT_PAGE foo
                                           // ENCBE: addiu $2, $3, %got_page(foo) # encoding: [0x24,0x62,A,A]
                                           // ENCLE: addiu $2, $3, %got_page(foo) # encoding: [A,A,0x62,0x24]
                                           // FIXUP: # fixup A - offset: 0, value: %got_page(foo), kind: fixup_Mips_GOT_PAGE

        addiu $2, $3, %got_ofst(foo)       // RELOC: R_MIPS_GOT_OFST foo
                                           // ENCBE: addiu $2, $3, %got_ofst(foo) # encoding: [0x24,0x62,A,A]
                                           // ENCLE: addiu $2, $3, %got_ofst(foo) # encoding: [A,A,0x62,0x24]
                                           // FIXUP: # fixup A - offset: 0, value: %got_ofst(foo), kind: fixup_Mips_GOT_OFST

        addiu $2, $3, %got_hi(foo)         // RELOC: R_MIPS_GOT_HI16 foo
                                           // ENCBE: addiu $2, $3, %got_hi(foo) # encoding: [0x24,0x62,A,A]
                                           // ENCLE: addiu $2, $3, %got_hi(foo) # encoding: [A,A,0x62,0x24]
                                           // FIXUP: # fixup A - offset: 0, value: %got_hi(foo), kind: fixup_Mips_GOT_HI16

        addiu $2, $3, %got_lo(foo)         // RELOC: R_MIPS_GOT_LO16 foo
                                           // ENCBE: addiu $2, $3, %got_lo(foo) # encoding: [0x24,0x62,A,A]
                                           // ENCLE: addiu $2, $3, %got_lo(foo) # encoding: [A,A,0x62,0x24]
                                           // FIXUP: # fixup A - offset: 0, value: %got_lo(foo), kind: fixup_Mips_GOT_LO16

//      addiu $2, $3, %neg(foo)            // FIXME: R_MIPS_SUB foo
                                           // ?????: R_MIPS_INSERT_A
                                           // ?????: R_MIPS_INSERT_B
                                           // ?????: R_MIPS_DELETE

        .set mips64
        daddiu $2, $3, %higher(foo)        // RELOC: R_MIPS_HIGHER foo
                                           // ENCBE: daddiu $2, $3, %higher(foo) # encoding: [0x64,0x62,A,A]
                                           // ENCLE: daddiu $2, $3, %higher(foo) # encoding: [A,A,0x62,0x64]
                                           // FIXUP: # fixup A - offset: 0, value: %higher(foo), kind: fixup_Mips_HIGHER

        daddiu $2, $3, %highest(foo)       // RELOC: R_MIPS_HIGHEST foo
                                           // ENCBE: daddiu $2, $3, %highest(foo) # encoding: [0x64,0x62,A,A]
                                           // ENCLE: daddiu $2, $3, %highest(foo) # encoding: [A,A,0x62,0x64]
                                           // FIXUP: # fixup A - offset: 0, value: %highest(foo), kind: fixup_Mips_HIGHEST

        .set mips0
        addiu $2, $3, %call_hi(foo)        // RELOC: R_MIPS_CALL_HI16 foo
                                           // ENCBE: addiu $2, $3, %call_hi(foo) # encoding: [0x24,0x62,A,A]
                                           // ENCLE: addiu $2, $3, %call_hi(foo) # encoding: [A,A,0x62,0x24]
                                           // FIXUP: # fixup A - offset: 0, value: %call_hi(foo), kind: fixup_Mips_CALL_HI16

        addiu $2, $3, %call_lo(foo)        // RELOC: R_MIPS_CALL_LO16 foo
                                           // ENCBE: addiu $2, $3, %call_lo(foo) # encoding: [0x24,0x62,A,A]
                                           // ENCLE: addiu $2, $3, %call_lo(foo) # encoding: [A,A,0x62,0x24]
                                           // FIXUP: # fixup A - offset: 0, value: %call_lo(foo), kind: fixup_Mips_CALL_LO16

                                           // ?????: R_MIPS_SCN_DISP foo
                                           // ?????: R_MIPS_REL16 foo
                                           // ?????: R_MIPS_ADD_IMMEDIATE foo
                                           // ?????: R_MIPS_PJUMP foo
                                           // ?????: R_MIPS_RELGOT foo
//      jalr $25                           // ?????: R_MIPS_JALR foo

                                           // ?????: R_MIPS_TLS_DTPMOD32 foo
//      .dtprelword foo                    // FIXME: R_MIPS_TLS_DTPREL32 foo
                                           // ?????: R_MIPS_TLS_DTPMOD64 foo
//      .dtpreldword foo                   // FIXME: R_MIPS_TLS_DTPREL64 foo
        addiu $2, $3, %tlsgd(foo)          // RELOC: R_MIPS_TLS_GD foo
                                           // ENCBE: addiu $2, $3, %tlsgd(foo) # encoding: [0x24,0x62,A,A]
                                           // ENCLE: addiu $2, $3, %tlsgd(foo) # encoding: [A,A,0x62,0x24]
                                           // FIXUP: # fixup A - offset: 0, value: %tlsgd(foo), kind: fixup_Mips_TLSGD

        addiu $2, $3, %tlsldm(foo)         // RELOC: R_MIPS_TLS_LDM foo
                                           // ENCBE: addiu $2, $3, %tlsldm(foo) # encoding: [0x24,0x62,A,A]
                                           // ENCLE: addiu $2, $3, %tlsldm(foo) # encoding: [A,A,0x62,0x24]
                                           // FIXUP: # fixup A - offset: 0, value: %tlsldm(foo), kind: fixup_Mips_TLSLDM

        addiu $2, $3, %dtprel_hi(foo)      // RELOC: R_MIPS_TLS_DTPREL_HI16 foo
                                           // ENCBE: addiu $2, $3, %dtprel_hi(foo) # encoding: [0x24,0x62,A,A]
                                           // ENCLE: addiu $2, $3, %dtprel_hi(foo) # encoding: [A,A,0x62,0x24]
                                           // FIXUP: # fixup A - offset: 0, value: %dtprel_hi(foo), kind: fixup_Mips_DTPREL_HI

        addiu $2, $3, %dtprel_lo(foo)      // RELOC: R_MIPS_TLS_DTPREL_LO16 foo
                                           // ENCBE: addiu $2, $3, %dtprel_lo(foo) # encoding: [0x24,0x62,A,A]
                                           // ENCLE: addiu $2, $3, %dtprel_lo(foo) # encoding: [A,A,0x62,0x24]
                                           // FIXUP: # fixup A - offset: 0, value: %dtprel_lo(foo), kind: fixup_Mips_DTPREL_LO

        addiu $2, $3, %gottprel(foo)       // RELOC: R_MIPS_TLS_GOTTPREL foo
                                           // ENCBE: addiu $2, $3, %gottprel(foo) # encoding: [0x24,0x62,A,A]
                                           // ENCLE: addiu $2, $3, %gottprel(foo) # encoding: [A,A,0x62,0x24]
                                           // FIXUP: # fixup A - offset: 0, value: %gottprel(foo), kind: fixup_Mips_GOTTPREL

//      .tprelword foo                     // FIXME: R_MIPS_TLS_TPREL32 foo
//      .tpreldword foo                    // FIXME: R_MIPS_TLS_TPREL64 foo
        addiu $2, $3, %tprel_hi(foo)       // RELOC: R_MIPS_TLS_TPREL_HI16 foo
                                           // ENCBE: addiu $2, $3, %tprel_hi(foo) # encoding: [0x24,0x62,A,A]
                                           // ENCLE: addiu $2, $3, %tprel_hi(foo) # encoding: [A,A,0x62,0x24]
                                           // FIXUP: # fixup A - offset: 0, value: %tprel_hi(foo), kind: fixup_Mips_TPREL_HI

        addiu $2, $3, %tprel_lo(foo)       // RELOC: R_MIPS_TLS_TPREL_LO16 foo
                                           // ENCBE: addiu $2, $3, %tprel_lo(foo) # encoding: [0x24,0x62,A,A]
                                           // ENCLE: addiu $2, $3, %tprel_lo(foo) # encoding: [A,A,0x62,0x24]
                                           // FIXUP: # fixup A - offset: 0, value: %tprel_lo(foo), kind: fixup_Mips_TPREL_LO

                                           // ?????: R_MIPS_GLOB_DAT foo
        .set mips32r6
        beqzc $2, foo                      // RELOC: R_MIPS_PC21_S2 foo
                                           // ENCBE: beqzc $2, foo # encoding: [0xd8,0b010AAAAA,A,A]
                                           // ENCLE: beqzc $2, foo # encoding: [A,A,0b010AAAAA,0xd8]
                                           // FIXUP: # fixup A - offset: 0, value: foo-4, kind: fixup_MIPS_PC21_S2

        bc foo                             // RELOC: R_MIPS_PC26_S2 foo
                                           // ENCBE: bc foo # encoding: [0b110010AA,A,A,A]
                                           // ENCLE: bc foo # encoding: [A,A,A,0b110010AA]
                                           // FIXUP: # fixup A - offset: 0, value: foo-4, kind: fixup_MIPS_PC26_S2

        .set mips64r6
        ldpc $2, foo                       // RELOC: R_MIPS_PC18_S3 foo
                                           // ENCBE: ldpc $2, foo # encoding: [0xec,0b010110AA,A,A]
                                           // ENCLE: ldpc $2, foo # encoding: [A,A,0b010110AA,0xec]
                                           // FIXUP: # fixup A - offset: 0, value: foo, kind: fixup_Mips_PC18_S3

        .set mips32r6
        lwpc $2, foo                       // RELOC: R_MIPS_PC19_S2 foo
                                           // ENCBE: lwpc $2, foo # encoding: [0xec,0b01001AAA,A,A]
                                           // ENCLE: lwpc $2, foo # encoding: [A,A,0b01001AAA,0xec]
                                           // FIXUP: # fixup A - offset: 0, value: foo, kind: fixup_MIPS_PC19_S2

        addiu $2, $3, %pcrel_hi(foo)       // RELOC: R_MIPS_PCHI16 foo
                                           // ENCBE: addiu $2, $3, %pcrel_hi(foo) # encoding: [0x24,0x62,A,A]
                                           // ENCLE: addiu $2, $3, %pcrel_hi(foo) # encoding: [A,A,0x62,0x24]
                                           // FIXUP: # fixup A - offset: 0, value: %pcrel_hi(foo), kind: fixup_MIPS_PCHI16

        addiu $2, $3, %pcrel_lo(foo)       // RELOC: R_MIPS_PCLO16 foo
                                           // ENCBE: addiu $2, $3, %pcrel_lo(foo) # encoding: [0x24,0x62,A,A]
                                           // ENCLE: addiu $2, $3, %pcrel_lo(foo) # encoding: [A,A,0x62,0x24]
                                           // FIXUP: # fixup A - offset: 0, value: %pcrel_lo(foo), kind: fixup_MIPS_PCLO16

        .set mips0
                                           // FIXME: R_MIPS16_*
                                           // ?????: R_MIPS_COPY foo
                                           // ?????: R_MIPS_JUMP_SLOT foo
                                           // FIXME: R_MICROMIPS_*
        .long foo-.                        // RELOC: R_MIPS_PC32 foo
//      .ehword foo                        // FIXME: R_MIPS_EH foo
