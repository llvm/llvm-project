# RUN: llvm-mc -triple=hexagon -filetype=obj %s -o - \
# RUN:   | llvm-objdump -r - | FileCheck %s

# Test coverage for HexagonELFObjectWriter: exercise TLS GD and LD
# relocation types.

# CHECK: R_HEX_GD_GOT
{
  r0 = memw(r0 + ##tls_gd_var@GDGOT)
}

# CHECK: R_HEX_IE_GOT
{
  r1 = memw(r1 + ##tls_ie_var@IEGOT)
}

# CHECK: R_HEX_TPREL
{
  r2 = memw(r2 + ##tls_le_var@TPREL)
}

# CHECK: R_HEX_DTPREL
{
  r3 = memw(r3 + ##tls_ld_var@DTPREL)
}
