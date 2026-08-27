# RUN: llvm-mc -triple=hexagon -filetype=obj %s -o - \
# RUN:   | llvm-objdump -r - | FileCheck %s

# Test coverage for HexagonELFObjectWriter: exercise FK_Data_1, FK_Data_2,
# FK_Data_4, and FK_Data_8 relocation sizes with GOT and TPREL variants.

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

## FK_Data_8: Hexagon is a 32-bit target -- 8-byte data fixups are
## lowered to R_HEX_32 (the address occupies the low 4 bytes on LE).
# CHECK: R_HEX_32
.8byte ext_8byte

# CHECK: R_HEX_32
.quad ext_quad
