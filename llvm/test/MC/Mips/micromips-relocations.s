# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding \
# RUN: -mattr=micromips | FileCheck %s -check-prefix=CHECK-FIXUP
# RUN: llvm-mc %s -filetype=obj -triple=mipsel-unknown-linux \
# RUN: -mattr=micromips | llvm-readobj -r - \
# RUN: | FileCheck %s -check-prefix=CHECK-ELF
#------------------------------------------------------------------------------
# Check that the assembler can handle the documented syntax
# for relocations.
#------------------------------------------------------------------------------
# CHECK-FIXUP: lui $2, %hi(_gp_disp)
# CHECK-FIXUP:        # encoding: [0xa2'A',0x41'A',0x00,0x00]
# CHECK-FIXUP:        # fixup A - offset: 0,
# CHECK-FIXUP:          value: %hi(_gp_disp),
# CHECK-FIXUP:          kind: fixup_MICROMIPS_HI16
# CHECK-FIXUP: addiu $2, $2, %lo(_gp_disp)
# CHECK-FIXUP:        # encoding: [0x42'A',0x30'A',0x00,0x00]
# CHECK-FIXUP:        # fixup A - offset: 0,
# CHECK-FIXUP:          value: %lo(_gp_disp),
# CHECK-FIXUP:          kind: fixup_MICROMIPS_LO16
# CHECK-FIXUP: lw $25, %call16(strchr)($gp)
# CHECK-FIXUP:        # encoding: [0x3c'A',0xff'A',0x00,0x00]
# CHECK-FIXUP:        # fixup A - offset: 0,
# CHECK-FIXUP:          value: %call16(strchr),
# CHECK-FIXUP:          kind: fixup_MICROMIPS_CALL16
# CHECK-FIXUP: lw $3, %got(loop_1)($2)
# CHECK-FIXUP:        # encoding: [0x62'A',0xfc'A',0x00,0x00]
# CHECK-FIXUP:        # fixup A - offset: 0,
# CHECK-FIXUP:          value: %got(loop_1),
# CHECK-FIXUP:          kind: fixup_MICROMIPS_GOT16
# CHECK-FIXUP: lui $2, %dtprel_hi(_gp_disp)
# CHECK-FIXUP:        # encoding: [0xa2'A',0x41'A',0x00,0x00]
# CHECK-FIXUP:        # fixup A - offset: 0,
# CHECK-FIXUP:          value: %dtprel_hi(_gp_disp),
# CHECK-FIXUP:          kind: fixup_MICROMIPS_TLS_DTPREL_HI16
# CHECK-FIXUP: addiu $2, $2, %dtprel_lo(_gp_disp)
# CHECK-FIXUP:        # encoding: [0x42'A',0x30'A',0x00,0x00]
# CHECK-FIXUP:        # fixup A - offset: 0,
# CHECK-FIXUP:          value: %dtprel_lo(_gp_disp),
# CHECK-FIXUP:          kind: fixup_MICROMIPS_TLS_DTPREL_LO16
# CHECK-FIXUP: lw $3, %got(loop_1)($2)
# CHECK-FIXUP:        # encoding: [0x62'A',0xfc'A',0x00,0x00]
# CHECK-FIXUP:        # fixup A - offset: 0,
# CHECK-FIXUP:          value: %got(loop_1),
# CHECK-FIXUP:          kind: fixup_MICROMIPS_GOT16
# CHECK-FIXUP: lw $4, %got_disp(loop_2)($3)
# CHECK-FIXUP:        # encoding: [0x83'A',0xfc'A',0x00,0x00]
# CHECK-FIXUP:        # fixup A - offset: 0,
# CHECK-FIXUP:          value: %got_disp(loop_2),
# CHECK-FIXUP:          kind: fixup_MICROMIPS_GOT_DISP
# CHECK-FIXUP: lw $5, %got_page(loop_3)($4)
# CHECK-FIXUP:        # encoding: [0xa4'A',0xfc'A',0x00,0x00]
# CHECK-FIXUP:        # fixup A - offset: 0,
# CHECK-FIXUP:          value: %got_page(loop_3),
# CHECK-FIXUP:          kind: fixup_MICROMIPS_GOT_PAGE
# CHECK-FIXUP: lw $6, %got_ofst(loop_4)($5)
# CHECK-FIXUP:        # encoding: [0xc5'A',0xfc'A',0x00,0x00]
# CHECK-FIXUP:        # fixup A - offset: 0,
# CHECK-FIXUP:          value: %got_ofst(loop_4),
# CHECK-FIXUP:          kind: fixup_MICROMIPS_GOT_OFST
# CHECK-FIXUP: lui $2, %tprel_hi(_gp_disp)
# CHECK-FIXUP:        # encoding: [0xa2'A',0x41'A',0x00,0x00]
# CHECK-FIXUP:        # fixup A - offset: 0,
# CHECK-FIXUP:          value: %tprel_hi(_gp_disp),
# CHECK-FIXUP:          kind: fixup_MICROMIPS_TLS_TPREL_HI16
# CHECK-FIXUP: addiu $2, $2, %tprel_lo(_gp_disp)
# CHECK-FIXUP:        # encoding: [0x42'A',0x30'A',0x00,0x00]
# CHECK-FIXUP:        # fixup A - offset: 0,
# CHECK-FIXUP:          value: %tprel_lo(_gp_disp),
# CHECK-FIXUP:          kind: fixup_MICROMIPS_TLS_TPREL_LO16
# CHECK-FIXUP: addiu $4, $gp, %tlsgd(a)
# CHECK-FIXUP:        # encoding: [0x9c'A',0x30'A',0x00,0x00]
# CHECK-FIXUP:        # fixup A - offset: 0,
# CHECK-FIXUP:          value: %tlsgd(a), kind: fixup_MICROMIPS_TLS_GD
# CHECK-FIXUP: addiu $4, $gp, %tlsldm(f.i)
# CHECK-FIXUP:        # encoding: [0x9c'A',0x30'A',0x00,0x00]
# CHECK-FIXUP:        # fixup A - offset: 0,
# CHECK-FIXUP:          value: %tlsldm(f.i), kind: fixup_MICROMIPS_TLS_LDM
#------------------------------------------------------------------------------
# Check that the appropriate relocations were created.
#------------------------------------------------------------------------------
# CHECK-ELF: Relocations [
# CHECK-ELF:     0x{{[0-9,A-F]+}} R_MICROMIPS_HI16
# CHECK-ELF:     0x{{[0-9,A-F]+}} R_MICROMIPS_LO16
# CHECK-ELF:     0x{{[0-9,A-F]+}} R_MICROMIPS_CALL16
# CHECK-ELF:     0x{{[0-9,A-F]+}} R_MICROMIPS_GOT16
# CHECK-ELF:     0x{{[0-9,A-F]+}} R_MICROMIPS_TLS_DTPREL_HI16
# CHECK-ELF:     0x{{[0-9,A-F]+}} R_MICROMIPS_TLS_DTPREL_LO16
# CHECK-ELF:     0x{{[0-9,A-F]+}} R_MICROMIPS_GOT16
# CHECK-ELF:     0x{{[0-9,A-F]+}} R_MICROMIPS_GOT_DISP
# CHECK-ELF:     0x{{[0-9,A-F]+}} R_MICROMIPS_GOT_PAGE
# CHECK-ELF:     0x{{[0-9,A-F]+}} R_MICROMIPS_GOT_OFST
# CHECK-ELF:     0x{{[0-9,A-F]+}} R_MICROMIPS_TLS_TPREL_HI16
# CHECK-ELF:     0x{{[0-9,A-F]+}} R_MICROMIPS_TLS_TPREL_LO16
# CHECK-ELF:     0x{{[0-9,A-F]+}} R_MICROMIPS_TLS_GD
# CHECK-ELF:     0x{{[0-9,A-F]+}} R_MICROMIPS_TLS_LDM
# CHECK-ELF: ]

    lui    $2, %hi(_gp_disp)
    addiu  $2, $2, %lo(_gp_disp)
    lw     $25, %call16(strchr)($gp)
    lw     $3, %got(loop_1)($2)
    lui    $2, %dtprel_hi(_gp_disp)
    addiu  $2, $2, %dtprel_lo(_gp_disp)
    lw     $3, %got(loop_1)($2)
    lw     $4, %got_disp(loop_2)($3)
    lw     $5, %got_page(loop_3)($4)
    lw     $6, %got_ofst(loop_4)($5)
    lui    $2, %tprel_hi(_gp_disp)
    addiu  $2, $2, %tprel_lo(_gp_disp)
    addiu  $4, $gp, %tlsgd(a)
    addiu  $4, $gp, %tlsldm(f.i)
