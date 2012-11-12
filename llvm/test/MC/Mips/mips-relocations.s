# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips32r2 | FileCheck %s
# Check that the assembler can handle the documented syntax
# for relocations.
# CHECK: .section __TEXT,__text,regular,pure_instructions
# CHECK:  lui   $2, %hi(_gp_disp)     # encoding: [A,A,0x02,0x3c]
# CHECK:                              #   fixup A - offset: 0, value: _gp_disp@ABS_HI, kind: fixup_Mips_HI16
# CHECK:  addiu $2, $2, %lo(_gp_disp) # encoding: [A,A,0x42,0x24]
# CHECK:                              #   fixup A - offset: 0, value: _gp_disp@ABS_LO, kind: fixup_Mips_LO16
# CHECK:  lw    $25, %call16(strchr)($gp)   # encoding: [A,A,0x99,0x8f]
# CHECK:                                    #   fixup A - offset: 0, value: strchr@GOT_CALL, kind: fixup_Mips_CALL16
# CHECK:  lw      $3, %got(loop_1)($2)    # encoding: [A,A,0x43,0x8c]
# CHECK:                                  #   fixup A - offset: 0, value: loop_1@GOT, kind: fixup_Mips_GOT_Local
# CHECK:  lui     $2, %dtprel_hi(_gp_disp) # encoding: [A,A,0x02,0x3c]
# CHECK:                                        #   fixup A - offset: 0, value: _gp_disp@DTPREL_HI, kind: fixup_Mips_DTPREL_HI
# CHECK:  addiu   $2, $2, %dtprel_hi(_gp_disp) # encoding: [A,A,0x42,0x24]
# CHECK:                                  #   fixup A - offset: 0, value: _gp_disp@DTPREL_HI, kind: fixup_Mips_DTPREL_HI
# CHECK:  lw      $3, %got(loop_1)($2)      # encoding: [A,A,0x43,0x8c]
# CHECK:                                    #   fixup A - offset: 0, value: loop_1@GOT, kind: fixup_Mips_GOT_Local
# CHECK:  lw      $4, %got_disp(loop_2)($3) # encoding: [A,A,0x64,0x8c]
# CHECK:                                    #   fixup A - offset: 0, value: loop_2@GOT_DISP, kind: fixup_Mips_GOT_DISP
# CHECK:  lw      $5, %got_page(loop_3)($4) # encoding: [A,A,0x85,0x8c]
# CHECK:                                    #   fixup A - offset: 0, value: loop_3@GOT_PAGE, kind: fixup_Mips_GOT_PAGE
# CHECK:  lw      $6, %got_ofst(loop_4)($5) # encoding: [A,A,0xa6,0x8c]
# CHECK:                                    #   fixup A - offset: 0, value: loop_4@GOT_OFST, kind: fixup_Mips_GOT_OFST
# CHECK:  lui     $2, %tprel_hi(_gp_disp)   # encoding: [A,A,0x02,0x3c]
# CHECK:                                    #   fixup A - offset: 0, value: _gp_disp@TPREL_HI, kind: fixup_Mips_TPREL_HI
# CHECK:  addiu   $2, $2, %tprel_lo(_gp_disp) # encoding: [A,A,0x42,0x24]
# CHECK:                                      #   fixup A - offset: 0, value: _gp_disp@TPREL_LO, kind: fixup_Mips_TPREL_LO

    lui	$2, %hi(_gp_disp)
	  addiu	$2, $2, %lo(_gp_disp)
    lw	$25, %call16(strchr)($gp)
    lw      $3, %got(loop_1)($2)
    lui	$2, %dtprel_hi(_gp_disp)
	  addiu	$2, $2, %dtprel_hi(_gp_disp)
    lw	$3, %got(loop_1)($2)
    lw	$4, %got_disp(loop_2)($3)
    lw	$5, %got_page(loop_3)($4)
    lw	$6, %got_ofst(loop_4)($5)
    lui	$2, %tprel_hi(_gp_disp)
	  addiu	$2, $2, %tprel_lo(_gp_disp)
