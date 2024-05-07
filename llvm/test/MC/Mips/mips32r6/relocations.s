# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r6 \
# RUN:   | FileCheck %s -check-prefix=CHECK-FIXUP
# RUN: llvm-mc %s -filetype=obj -triple=mips-unknown-linux -mcpu=mips32r6 \
# RUN:   | llvm-readobj -r - | FileCheck %s -check-prefix=CHECK-ELF
#------------------------------------------------------------------------------
# Check that the assembler can handle the documented syntax for fixups.
#------------------------------------------------------------------------------
# CHECK-FIXUP: lapc $2, bar     # encoding: [0xec,0b01000AAA,A,A]
# CHECK-FIXUP:                  # fixup A - offset: 0,
# CHECK-FIXUP:                    value: bar, kind: fixup_MIPS_PC19_S2
# CHECK-FIXUP: beqc $5, $6, bar # encoding: [0x20,0xa6,A,A]
# CHECK-FIXUP:                  #   fixup A - offset: 0,
# CHECK-FIXUP:                      value: bar-4, kind: fixup_Mips_PC16
# CHECK-FIXUP: bnec $5, $6, bar # encoding: [0x60,0xa6,A,A]
# CHECK-FIXUP:                  #   fixup A - offset: 0,
# CHECK-FIXUP:                      value: bar-4, kind: fixup_Mips_PC16
# CHECK-FIXUP: beqzc $9, bar    # encoding: [0xd9,0b001AAAAA,A,A]
# CHECK-FIXUP:                  #   fixup A - offset: 0,
# CHECK-FIXUP:                      value: bar-4, kind: fixup_MIPS_PC21_S2
# CHECK-FIXUP: bnezc $9, bar    # encoding: [0xf9,0b001AAAAA,A,A]
# CHECK-FIXUP:                  #   fixup A - offset: 0,
# CHECK-FIXUP:                      value: bar-4, kind: fixup_MIPS_PC21_S2
# CHECK-FIXUP: balc  bar        # encoding: [0b111010AA,A,A,A]
# CHECK-FIXUP:                  #   fixup A - offset: 0,
# CHECK-FIXUP:                      value: bar-4, kind: fixup_MIPS_PC26_S2
# CHECK-FIXUP: bc    bar        # encoding: [0b110010AA,A,A,A]
# CHECK-FIXUP:                  #   fixup A - offset: 0,
# CHECK-FIXUP:                      value: bar-4, kind: fixup_MIPS_PC26_S2
# CHECK-FIXUP: aluipc $2, %pcrel_hi(bar)    # encoding: [0xec,0x5f,A,A]
# CHECK-FIXUP:                              #   fixup A - offset: 0,
# CHECK-FIXUP:                                  value: %pcrel_hi(bar),
# CHECK-FIXUP:                                  kind: fixup_MIPS_PCHI16
# CHECK-FIXUP: addiu $2, $2, %pcrel_lo(bar) # encoding: [0x24,0x42,A,A]
# CHECK-FIXUP:                              #   fixup A - offset: 0,
# CHECK-FIXUP:                                  value: %pcrel_lo(bar),
# CHECK-FIXUP:                                  kind: fixup_MIPS_PCLO16
# CHECK-FIXUP: lapc $2, bar     # encoding: [0xec,0b01000AAA,A,A]
# CHECK-FIXUP:                  # fixup A - offset: 0,
# CHECK-FIXUP:                    value: bar, kind: fixup_MIPS_PC19_S2
# CHECK-FIXUP: lwpc    $2, bar  # encoding: [0xec,0b01001AAA,A,A]
# CHECK-FIXUP:                  #   fixup A - offset: 0,
# CHECK-FIXUP:                      value: bar, kind: fixup_MIPS_PC19_S2
# CHECK-FIXUP: jialc   $5, bar  # encoding: [0xf8,0x05,A,A]
# CHECK-FIXUP:                  #   fixup A - offset: 0,
# CHECK-FIXUP:                      value: bar, kind: fixup_Mips_LO16
# CHECK-FIXUP: jic     $5, bar  # encoding: [0xd8,0x05,A,A]
# CHECK-FIXUP:                  #   fixup A - offset: 0,
# CHECK-FIXUP:                      value: bar, kind: fixup_Mips_LO16
#------------------------------------------------------------------------------
# Check that the appropriate relocations were created.
#------------------------------------------------------------------------------
# CHECK-ELF: Relocations [
# CHECK-ELF:     0x0 R_MIPS_PC19_S2 bar
# CHECK-ELF:     0x4 R_MIPS_PC16 bar
# CHECK-ELF:     0xC R_MIPS_PC16 bar
# CHECK-ELF:     0x14 R_MIPS_PC21_S2 bar
# CHECK-ELF:     0x1C R_MIPS_PC21_S2 bar
# CHECK-ELF:     0x24 R_MIPS_PC26_S2 bar
# CHECK-ELF:     0x28 R_MIPS_PC26_S2 bar
# CHECK-ELF:     0x2C R_MIPS_PCHI16 bar
# CHECK-ELF:     0x30 R_MIPS_PCLO16 bar
# CHECK-ELF:     0x34 R_MIPS_PC19_S2 bar
# CHECK-ELF:     0x38 R_MIPS_PC19_S2 bar
# CHECK-ELF:     0x3C R_MIPS_LO16 bar
# CHECK-ELF:     0x40 R_MIPS_LO16 bar
# CHECK-ELF: ]

  addiupc   $2,bar
  beqc  $5, $6, bar
  bnec  $5, $6, bar
  beqzc $9, bar
  bnezc $9, bar
  balc  bar
  bc    bar
  aluipc $2, %pcrel_hi(bar)
  addiu  $2, $2, %pcrel_lo(bar)
  lapc      $2,bar
  lwpc      $2,bar
  jialc  $5, bar
  jic    $5, bar
