# RUN: llvm-mc -triple=hexagon -filetype=obj -o - %s | llvm-objdump -d - | FileCheck %s
# Hexagon Programmer's Reference Manual 11.6 MEMOP

# Operation on memory byte
# CHECK: 95 d9 11 3e
memb(r17+#51) += r21
# CHECK: b5 d9 11 3e
memb(r17+#51) -= r21
# CHECK: d5 d9 11 3e
memb(r17+#51) &= r21
# CHECK: f5 d9 11 3e
memb(r17+#51) |= r21
# CHECK: 95 d9 11 3f
memb(r17+#51) += #21
# CHECK: b5 d9 11 3f
memb(r17+#51) -= #21
# CHECK: d5 d9 11 3f
memb(r17+#51) = clrbit(#21)
# CHECK: f5 d9 11 3f
memb(r17+#51) = setbit(#21)

# Operation on memory halfword
# CHECK: 95 d9 31 3e
memh(r17+#102) += r21
# CHECK: b5 d9 31 3e
memh(r17+#102) -= r21
# CHECK: d5 d9 31 3e
memh(r17+#102) &= r21
# CHECK: f5 d9 31 3e
memh(r17+#102) |= r21
# CHECK: 95 d9 31 3f
memh(r17+#102) += #21
# CHECK: b5 d9 31 3f
memh(r17+#102) -= #21
# CHECK: d5 d9 31 3f
memh(r17+#102) = clrbit(#21)
# CHECK: f5 d9 31 3f
memh(r17+#102) = setbit(#21)

# Operation on memory word
# CHECK: 95 d9 51 3e
memw(r17+#204) += r21
# CHECK: b5 d9 51 3e
memw(r17+#204) -= r21
# CHECK: d5 d9 51 3e
memw(r17+#204) &= r21
# CHECK: f5 d9 51 3e
memw(r17+#204) |= r21
# CHECK: 95 d9 51 3f
memw(r17+#204) += #21
# CHECK: b5 d9 51 3f
memw(r17+#204) -= #21
# CHECK: d5 d9 51 3f
memw(r17+#204) = clrbit(#21)
# CHECK: f5 d9 51 3f
memw(r17+#204) = setbit(#21)
