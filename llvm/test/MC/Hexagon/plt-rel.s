# RUN: llvm-mc -triple=hexagon -filetype=obj %s | llvm-objdump -d -r - | FileCheck %s

call foo@GDPLT
# CHECK: R_HEX_GD_PLT_B22_PCREL
call ##foo@GDPLT
# CHECK:  R_HEX_GD_PLT_B32_PCREL_X
# CHECK: R_HEX_GD_PLT_B22_PCREL_X

call foo@LDPLT
# CHECK:  R_HEX_LD_PLT_B22_PCREL
call ##foo@LDPLT
# CHECK:  R_HEX_LD_PLT_B32_PCREL_X
# CHECK:  R_HEX_LD_PLT_B22_PCREL_X
