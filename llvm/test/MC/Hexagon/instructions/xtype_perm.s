# RUN: llvm-mc -triple=hexagon -filetype=obj -o - %s | llvm-objdump -d - | FileCheck %s
# Hexagon Programmer's Reference Manual 11.10.6 XTYPE/PERM

# CABAC decode bin
# CHECK: d0 de d4 c1
r17:16 = decbin(r21:20, r31:30)

# Saturate
# CHECK: 11 c0 d4 88
r17 = sat(r21:20)
# CHECK: 91 c0 d5 8c
r17 = sath(r21)
# CHECK: b1 c0 d5 8c
r17 = satuh(r21)
# CHECK: d1 c0 d5 8c
r17 = satub(r21)
# CHECK: f1 c0 d5 8c
r17 = satb(r21)

# Swizzle bytes
# CHECK: f1 c0 95 8c
r17 = swiz(r21)

# Vector align
# CHECK: 70 d4 1e c2
r17:16 = valignb(r21:20, r31:30, p3)
# CHECK: 70 de 94 c2
r17:16 = vspliceb(r21:20, r31:30, p3)

# Vector round and pack
# CHECK: 91 c0 94 88
r17 = vrndwh(r21:20)
# CHECK: d1 c0 94 88
r17 = vrndwh(r21:20):sat

# Vector saturate and pack
# CHECK: 11 c0 14 88
r17 = vsathub(r21:20)
# CHECK: 51 c0 14 88
r17 = vsatwh(r21:20)
# CHECK: 91 c0 14 88
r17 = vsatwuh(r21:20)
# CHECK: d1 c0 14 88
r17 = vsathb(r21:20)
# CHECK: 11 c0 95 8c
r17 = vsathb(r21)
# CHECK: 51 c0 95 8c
r17 = vsathub(r21)

# Vector saturate without pack
# CHECK: 90 c0 14 80
r17:16 = vsathub(r21:20)
# CHECK: b0 c0 14 80
r17:16 = vsatwuh(r21:20)
# CHECK: d0 c0 14 80
r17:16 = vsatwh(r21:20)
# CHECK: f0 c0 14 80
r17:16 = vsathb(r21:20)

# Vector shuffle
# CHECK: 50 de 14 c1
r17:16 = shuffeb(r21:20, r31:30)
# CHECK: 90 d4 1e c1
r17:16 = shuffob(r21:20, r31:30)
# CHECK: d0 de 14 c1
r17:16 = shuffeh(r21:20, r31:30)
# CHECK: 10 d4 9e c1
r17:16 = shuffoh(r21:20, r31:30)

# Vector splat bytes
# CHECK: f1 c0 55 8c
r17 = vsplatb(r21)

# Vector splat halfwords
# CHECK: 50 c0 55 84
r17:16 = vsplath(r21)

# Vector splice
# CHECK: 70 de 94 c0
r17:16 = vspliceb(r21:20, r31:30, #3)
# CHECK: 70 de 94 c2
r17:16 = vspliceb(r21:20, r31:30, p3)

# Vector sign extend
# CHECK: 10 c0 15 84
r17:16 = vsxtbh(r21)
# CHECK: 90 c0 15 84
r17:16 = vsxthw(r21)

# Vector truncate
# CHECK: 11 c0 94 88
r17 = vtrunohb(r21:20)
# CHECK: 51 c0 94 88
r17 = vtrunehb(r21:20)
# CHECK: 50 de 94 c1
r17:16 = vtrunewh(r21:20, r31:30)
# CHECK: 90 de 94 c1
r17:16 = vtrunowh(r21:20, r31:30)

# Vector zero extend
# CHECK: 50 c0 15 84
r17:16 = vzxtbh(r21)
# CHECK: d0 c0 15 84
r17:16 = vzxthw(r21)
