# RUN: llvm-mc -arch=hexagon -filetype=obj %s | llvm-objdump -r - | FileCheck %s
#

# make sure the fixups emitted match what is
# expected.
.Lpc:
    r0 = add (pc, ##foo@PCREL)

# CHECK: R_HEX_B32_PCREL_X
# CHECK: R_HEX_6_PCREL_X

