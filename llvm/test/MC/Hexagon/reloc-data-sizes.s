# RUN: llvm-mc -triple=hexagon -filetype=obj %s -o - \
# RUN:   | llvm-objdump -r - | FileCheck %s

# Test coverage for HexagonELFObjectWriter: exercise FK_Data_1, FK_Data_2,
# and FK_Data_4 relocation sizes with GOT and TPREL variants.

.data
# CHECK: R_HEX_32
.word ext_sym

# CHECK: R_HEX_16
.short ext_sym16

# CHECK: R_HEX_8
.byte ext_sym8

# CHECK: R_HEX_GOT_32
.word ext_got@GOT

# CHECK: R_HEX_TPREL_32
.word ext_tprel@TPREL
