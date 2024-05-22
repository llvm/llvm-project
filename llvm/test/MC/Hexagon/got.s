# RUN: llvm-mc -triple=hexagon -filetype=obj %s | llvm-objdump -r - | FileCheck %s
#

# make sure the fixups emitted match what is
# expected.
.Lgot:
    r0 = memw (r1 + ##foo@GOT)

# CHECK: R_HEX_GOT_32_6_X foo
# CHECK: R_HEX_GOT_11_X foo

