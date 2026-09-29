# RUN: llvm-mc -triple=hexagon -filetype=obj -o - %s | llvm-objdump -d - | FileCheck %s
# Hexagon Programmer's Reference Manual 11.10.2 XTYPE/BIT

# Count leading
# CHECK: 11 c0 54 88
r17 = clb(r21:20)
# CHECK: 51 c0 54 88
r17 = cl0(r21:20)
# CHECK: 91 c0 54 88
r17 = cl1(r21:20)
# CHECK: 11 c0 74 88
r17 = normamt(r21:20)
# CHECK: 51 d7 74 88
r17 = add(clb(r21:20), #23)
# CHECK: 11 d7 35 8c
r17 = add(clb(r21), #23)
# CHECK: 91 c0 15 8c
r17 = clb(r21)
# CHECK: b1 c0 15 8c
r17 = cl0(r21)
# CHECK: d1 c0 15 8c
r17 = cl1(r21)
# CHECK: f1 c0 15 8c
r17 = normamt(r21)

# Count population
# CHECK: 71 c0 74 88
r17 = popcount(r21:20)

# Count trailing
# CHECK: 51 c0 f4 88
r17 = ct0(r21:20)
# CHECK: 91 c0 f4 88
r17 = ct1(r21:20)
# CHECK: 91 c0 55 8c
r17 = ct0(r21)
# CHECK: b1 c0 55 8c
r17 = ct1(r21)

# Extract bitfield
# CHECK: f0 df 54 81
r17:16 = extractu(r21:20, #31, #23)
# CHECK: f0 df 54 8a
r17:16 = extract(r21:20, #31, #23)
# CHECK: f1 df 55 8d
r17 = extractu(r21, #31, #23)
# CHECK: f1 df d5 8d
r17 = extract(r21, #31, #23)
# CHECK: 10 de 14 c1
r17:16 = extractu(r21:20, r31:30)
# CHECK: 90 de d4 c1
r17:16 = extract(r21:20, r31:30)
# CHECK: 11 de 15 c9
r17 = extractu(r21, r31:30)
# CHECK: 51 de 15 c9
r17 = extract(r21, r31:30)

# Insert bitfield
# CHECK: f0 df 54 83
r17:16 = insert(r21:20, #31, #23)
# CHECK: f1 df 55 8f
r17 = insert(r21, #31, #23)
# CHECK: 11 de 15 c8
r17 = insert(r21, r31:30)
# CHECK: 10 de 14 ca
r17:16 = insert(r21:20, r31:30)

# Interleave/deinterleave
# CHECK: 90 c0 d4 80
r17:16 = deinterleave(r21:20)
# CHECK: b0 c0 d4 80
r17:16 = interleave(r21:20)

# Linear feedback-shift iteration
# CHECK: d0 de 94 c1
r17:16 = lfs(r21:20, r31:30)

# Masked parity
# CHECK: 11 de 14 d0
r17 = parity(r21:20, r31:30)
# CHECK: 11 df f5 d5
r17 = parity(r21, r31)

# Bit reverse
# CHECK: d0 c0 d4 80
r17:16 = brev(r21:20)
# CHECK: d1 c0 55 8c
r17 = brev(r21)

# Set/clear/toggle bit
# CHECK: 11 df d5 8c
r17 = setbit(r21, #31)
# CHECK: 31 df d5 8c
r17 = clrbit(r21, #31)
# CHECK: 51 df d5 8c
r17 = togglebit(r21, #31)
# CHECK: 11 df 95 c6
r17 = setbit(r21, r31)
# CHECK: 51 df 95 c6
r17 = clrbit(r21, r31)
# CHECK: 91 df 95 c6
r17 = togglebit(r21, r31)

# Split bitfield
# CHECK: 90 df d5 88
r17:16 = bitsplit(r21, #31)
# CHECK: 10 df 35 d4
r17:16 = bitsplit(r21, r31)

# Table index
# CHECK: f1 cd 15 87
r17 = tableidxb(r21, #7, #13):raw
# CHECK: f1 cd 55 87
r17 = tableidxh(r21, #7, #13):raw
# CHECK: f1 cd 95 87
r17 = tableidxw(r21, #7, #13):raw
# CHECK: f1 cd d5 87
r17 = tableidxd(r21, #7, #13):raw
