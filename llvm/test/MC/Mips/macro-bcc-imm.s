# RUN: llvm-mc %s -triple=mips -mcpu=mips32r2 -show-encoding 2>&1 | \
# RUN:     FileCheck %s --check-prefix=ALL

    .text
foo:
    beql $a2, 0x1ffff, foo # ALL: lui $1, 1
                           # ALL: ori $1, $1, 65535
                           # ALL: beql  $6, $1, foo
                           # ALL:  #   fixup A - offset: 0, value: foo-4, kind: fixup_Mips_PC16
                           # ALL: nop
    beql $a2, -4096, foo   # ALL: addiu $1, $zero, -4096
                           # ALL: beql  $6, $1, foo
                           # ALL: #   fixup A - offset: 0, value: foo-4, kind: fixup_Mips_PC16
    beql $a2, -0x10000, foo # ALL: lui $1, 65535
                            # ALL: beql  $6, $1, foo
                            # ALL: # fixup A - offset: 0, value: foo-4, kind: fixup_Mips_PC16
    beql $a2, 16, foo     # ALL: addiu   $1, $zero, 16
                          # ALL: beql    $6, $1, foo
                          # ALL:  # fixup A - offset: 0, value: foo-4, kind: fixup_Mips_PC16
                          # ALL: nop
    bnel $a2, 0x1ffff, foo # ALL: lui $1, 1
                           # ALL: ori $1, $1, 65535
                           # ALL: bnel  $6, $1, foo
                           # ALL:  #   fixup A - offset: 0, value: foo-4, kind: fixup_Mips_PC16
                           # ALL: nop
    bnel $a2, -4096, foo   # ALL: addiu $1, $zero, -4096
                           # ALL: bnel  $6, $1, foo
                           # ALL: #   fixup A - offset: 0, value: foo-4, kind: fixup_Mips_PC16
    bnel $a2, -0x10000, foo # ALL: lui $1, 65535
                            # ALL: bnel  $6, $1, foo
                            # ALL: # fixup A - offset: 0, value: foo-4, kind: fixup_Mips_PC16
    bnel $a2, 16, foo     # ALL: addiu   $1, $zero, 16
                          # ALL: bnel    $6, $1, foo
                          # ALL: #   fixup A - offset: 0, value: foo-4, kind: fixup_Mips_PC16
                          # ALL: nop
    beql $a2, 32767, foo  # ALL: addiu   $1, $zero, 32767
                          # ALL: beql    $6, $1, foo
                          # ALL: #   fixup A - offset: 0, value: foo-4, kind: fixup_Mips_PC16
                          # ALL: nop
    bnel $a2, 32768, foo  # ALL: ori     $1, $zero, 32768
                          # ALL: bnel    $6, $1, foo
                          # ALL: #   fixup A - offset: 0, value: foo-4, kind: fixup_Mips_PC16
                          # ALL: nop
    blt $a2, 16, foo      # ALL: addiu $1, $zero, 16
                          # ALL: slt   $1, $6, $1
                          # ALL: bnez  $1, foo
                          # ALL: # fixup A - offset: 0, value: foo-4, kind: fixup_Mips_PC16
    ble $a2, 16, foo      # ALL: addiu $1, $zero, 16
                          # ALL: slt   $1, $1, $6
                          # ALL: beqz  $1, foo
                          # ALL: # fixup A - offset: 0, value: foo-4, kind: fixup_Mips_PC16
    bge $a2, 32767, foo   # ALL: addiu $1, $zero, 32767
                          # ALL: slt   $1, $6, $1
                          # ALL: beqz  $1, foo
                          # ALL: # fixup A - offset: 0, value: foo-4, kind: fixup_Mips_PC16
    bgt $a2, 32768, foo   # ALL: ori   $1, $zero, 32768
                          # ALL: slt   $1, $1, $6
                          # ALL: bnez  $1, foo
                          # ALL: # fixup A - offset: 0, value: foo-4, kind: fixup_Mips_PC16
    bltu $a2, 16, foo     # ALL: addiu $1, $zero, 16
                          # ALL: sltu  $1, $6, $1
                          # ALL: bnez  $1, foo
                          # ALL: # fixup A - offset: 0, value: foo-4, kind: fixup_Mips_PC16
    bleu $a2, 16, foo     # ALL: addiu $1, $zero, 16
                          # ALL: sltu  $1, $1, $6
                          # ALL: beqz  $1, foo
                          # ALL: # fixup A - offset: 0, value: foo-4, kind: fixup_Mips_PC16
    bgeu $a2, 32767, foo  # ALL: addiu $1, $zero, 32767
                          # ALL: sltu  $1, $6, $1
                          # ALL: beqz  $1, foo
                          # ALL: # fixup A - offset: 0, value: foo-4, kind: fixup_Mips_PC16
    bgtu $a2, 32768, foo  # ALL: ori   $1, $zero, 32768
                          # ALL: sltu  $1, $1, $6
                          # ALL: bnez  $1, foo
                          # ALL: # fixup A - offset: 0, value: foo-4, kind: fixup_Mips_PC16
    bltl $a2, 16, foo     # ALL: addiu $1, $zero, 16
                          # ALL: slt   $1, $6, $1
                          # ALL: bnezl $1, foo
                          # ALL: # fixup A - offset: 0, value: foo-4, kind: fixup_Mips_PC16
    blel $a2, 16, foo     # ALL: addiu $1, $zero, 16
                          # ALL: slt   $1, $1, $6
                          # ALL: beqzl $1, foo
                          # ALL: # fixup A - offset: 0, value: foo-4, kind: fixup_Mips_PC16
    bgel $a2, 32767, foo  # ALL: addiu $1, $zero, 32767
                          # ALL: slt   $1, $6, $1
                          # ALL: beqzl $1, foo
                          # ALL: # fixup A - offset: 0, value: foo-4, kind: fixup_Mips_PC16
    bgtl $a2, 32768, foo  # ALL: ori   $1, $zero, 32768
                          # ALL: slt   $1, $1, $6
                          # ALL: bnezl $1, foo
                          # ALL: # fixup A - offset: 0, value: foo-4, kind: fixup_Mips_PC16
    bltul $a2, 16, foo    # ALL: addiu $1, $zero, 16
                          # ALL: sltu  $1, $6, $1
                          # ALL: bnezl $1, foo
                          # ALL: # fixup A - offset: 0, value: foo-4, kind: fixup_Mips_PC16
    bleul $a2, 16, foo    # ALL: addiu $1, $zero, 16
                          # ALL: sltu  $1, $1, $6
                          # ALL: beqzl $1, foo
                          # ALL: # fixup A - offset: 0, value: foo-4, kind: fixup_Mips_PC16
    bgeul $a2, 32767, foo # ALL: addiu $1, $zero, 32767
                          # ALL: sltu  $1, $6, $1
                          # ALL: beqzl $1, foo
                          # ALL: # fixup A - offset: 0, value: foo-4, kind: fixup_Mips_PC16
    bgtul $a2, 65536, foo # ALL: lui   $1, 1
                          # ALL: sltu  $1, $1, $6
                          # ALL: bnezl $1, foo
                          # ALL: # fixup A - offset: 0, value: foo-4, kind: fixup_Mips_PC16
