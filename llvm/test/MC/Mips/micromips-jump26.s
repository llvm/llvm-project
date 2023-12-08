# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding \
# RUN: -mattr=micromips | FileCheck %s -check-prefix=CHECK-FIXUP
# RUN: llvm-mc %s -filetype=obj -triple=mipsel-unknown-linux \
# RUN: -mattr=micromips | llvm-readobj -r - \
# RUN: | FileCheck %s -check-prefix=CHECK-ELF
#------------------------------------------------------------------------------
# Check that the assembler can handle the documented syntax
# for relocations.
#------------------------------------------------------------------------------
# CHECK-FIXUP: foo:
# CHECK-FIXUP:   jal bar # encoding: [A,0xf4'A',A,0b000000AA]
# CHECK-FIXUP:           #   fixup A - offset: 0,
# CHECK-FIXUP:               value: bar, kind: fixup_MICROMIPS_26_S1
# CHECK-FIXUP:   nop     # encoding: [0x00,0x00,0x00,0x00]
#------------------------------------------------------------------------------
# Check that the appropriate relocations were created.
#------------------------------------------------------------------------------
# CHECK-ELF: Relocations [
# CHECK-ELF:     0x{{[0-9,A-F]+}} R_MICROMIPS_26_S1
# CHECK-ELF: ]

foo:
  jal bar
