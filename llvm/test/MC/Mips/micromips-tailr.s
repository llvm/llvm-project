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
# CHECK-FIXUP:   addiu $2, $zero, 1332
# CHECK-FIXUP:         # encoding: [0x40,0x30,0x34,0x05]
# CHECK-FIXUP:   j foo # encoding: [A,0xd4'A',A,0b000000AA]
# CHECK-FIXUP-NEXT:         #   fixup A - offset: 0, value: foo, kind: fixup_MICROMIPS_26_S1
# CHECK-FIXUP-NEXT:   nop   # encoding: [0x00,0x0c]
#------------------------------------------------------------------------------
# Check that the appropriate relocations were created.
#------------------------------------------------------------------------------
# CHECK-ELF: Relocations [
# CHECK-ELF:     0x{{[0-9,A-F]+}} R_MICROMIPS_26_S1
# CHECK-ELF: ]

foo:
  addiu $2, $0, 1332
  j foo
