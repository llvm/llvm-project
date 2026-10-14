# RUN: llvm-mc -triple=hexagon -filetype=obj %s -o - \
# RUN:   | llvm-objdump -r - | FileCheck %s

# Test coverage for HexagonELFObjectWriter: exercise various PIC and
# TLS relocation types through the getRelocType() paths.

# GOT-relative access for a global variable.
# CHECK: R_HEX_GOT_32_6_X
# CHECK: R_HEX_GOT_11_X
{
  r0 = memw(r0 + ##gvar@GOT)
}

# IE-GOT access for an initial-exec TLS variable.
# CHECK: R_HEX_IE_GOT_32_6_X
# CHECK: R_HEX_IE_GOT_11_X
{
  r1 = memw(r1 + ##tls_ie_var@IEGOT)
}

# TPREL access for a local-exec TLS variable.
# CHECK: R_HEX_TPREL_32_6_X
# CHECK: R_HEX_TPREL_11_X
{
  r2 = memw(r2 + ##tls_le_var@TPREL)
}

# PC-relative GOT base computation.
# CHECK: R_HEX_B32_PCREL_X
# CHECK: R_HEX_6_PCREL_X
{
  r3 = add(pc, ##_GLOBAL_OFFSET_TABLE_@PCREL)
}
