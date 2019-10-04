# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r6 \
# RUN:   -mattr=micromips | FileCheck %s -check-prefix=CHECK-FIXUP
# RUN: llvm-mc %s -filetype=obj -triple=mips-unknown-linux -mcpu=mips32r6 \
# RUN:   -mattr=micromips | llvm-readobj -r | FileCheck %s -check-prefix=CHECK-ELF
#------------------------------------------------------------------------------
# Check that the assembler can handle the documented syntax for fixups.
#------------------------------------------------------------------------------
# CHECK-FIXUP: balc  bar        # encoding: [0b101101AA,A,A,A]
# CHECK-FIXUP:                  #   fixup A - offset: 0,
# CHECK-FIXUP:                      value: bar-4, kind: fixup_MICROMIPS_PC26_S1
# CHECK-FIXUP: bc    bar        # encoding: [0b100101AA,A,A,A]
# CHECK-FIXUP:                  #   fixup A - offset: 0,
# CHECK-FIXUP:                      value: bar-4, kind: fixup_MICROMIPS_PC26_S1
# CHECK-FIXUP: lapc  $2, bar    # encoding: [0x78,0b01000AAA,A,A]
# CHECK-FIXUP:                  #   fixup A - offset: 0,
# CHECK-FIXUP:                      value: bar, kind: fixup_MICROMIPS_PC19_S2
# CHECK-FIXUP: lapc  $2, bar    # encoding: [0x78,0b01000AAA,A,A]
# CHECK-FIXUP:                  #   fixup A - offset: 0,
# CHECK-FIXUP:                      value: bar, kind: fixup_MICROMIPS_PC19_S2
# CHECK-FIXUP: lwpc    $2, bar  # encoding: [0x78,0b01001AAA,A,A]
# CHECK-FIXUP:                  #   fixup A - offset: 0,
# CHECK-FIXUP:                      value: bar, kind: fixup_MICROMIPS_PC19_S2
# CHECK-FIXUP: beqzc $3, bar    # encoding: [0x80,0b011AAAAA,A,A]
# CHECK-FIXUP:                  #   fixup A - offset: 0,
# CHECK-FIXUP:                      value: bar-4, kind: fixup_MICROMIPS_PC21_S1
# CHECK-FIXUP: bnezc $3, bar    # encoding: [0xa0,0b011AAAAA,A,A]
# CHECK-FIXUP:                  #   fixup A - offset: 0,
# CHECK-FIXUP:                      value: bar-4, kind: fixup_MICROMIPS_PC21_S1
# CHECK-FIXUP: jialc $5, bar    # encoding: [0x80,0x05,A,A]
# CHECK-FIXUP:                  #   fixup A - offset: 0,
# CHECK-FIXUP:                      value: bar, kind: fixup_MICROMIPS_LO16
# CHECK-FIXUP: jic   $5, bar    # encoding: [0xa0,0x05,A,A]
# CHECK-FIXUP:                  #   fixup A - offset: 0,
# CHECK-FIXUP:                      value: bar, kind: fixup_MICROMIPS_LO16
#------------------------------------------------------------------------------
# Check that the appropriate relocations were created.
#------------------------------------------------------------------------------
# CHECK-ELF: Relocations [
# CHECK-ELF:     0x0 R_MICROMIPS_PC26_S1 bar 0x0
# CHECK-ELF:     0x4 R_MICROMIPS_PC26_S1 bar 0x0
# CHECK-ELF:     0x8 R_MICROMIPS_PC19_S2 bar 0x0
# CHECK-ELF:     0xC R_MICROMIPS_PC19_S2 bar 0x0
# CHECK-ELF:     0x10 R_MICROMIPS_PC19_S2 bar 0x0
# CHECK-ELF:     0x14 R_MICROMIPS_PC21_S1 bar 0x0
# CHECK-ELF:     0x18 R_MICROMIPS_PC21_S1 bar 0x0
# CHECK-ELF:     0x1C R_MICROMIPS_LO16 bar 0x0
# CHECK-ELF:     0x20 R_MICROMIPS_LO16 bar 0x0
# CHECK-ELF: ]

  balc  bar
  bc    bar
  addiupc $2,bar
  lapc    $2,bar
  lwpc    $2,bar
  beqzc  $3, bar
  bnezc  $3, bar
  jialc  $5, bar
  jic    $5, bar
