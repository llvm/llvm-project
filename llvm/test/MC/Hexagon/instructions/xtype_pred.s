# RUN: llvm-mc -triple=hexagon -filetype=obj -o - %s | llvm-objdump -d - | FileCheck %s
# Hexagon Programmer's Reference Manual 11.10.7 XTYPE/PRED

# Bounds check
# CHECK: 83 f4 10 d2
p3 = boundscheck(r17:16, r21:20):raw:lo
# CHECK: a3 f4 10 d2
p3 = boundscheck(r17:16, r21:20):raw:hi

# Compare byte
# CHECK: 43 d5 d1 c7
p3 = cmpb.gt(r17, r21)
# CHECK: c3 d5 d1 c7
p3 = cmpb.eq(r17, r21)
# CHECK: e3 d5 d1 c7
p3 = cmpb.gtu(r17, r21)
# CHECK: a3 c2 11 dd
p3 = cmpb.eq(r17, #21)
# CHECK: a3 c2 31 dd
p3 = cmpb.gt(r17, #21)
# CHECK: a3 c2 51 dd
p3 = cmpb.gtu(r17, #21)

# Compare half
# CHECK: 63 d5 d1 c7
p3 = cmph.eq(r17, r21)
# CHECK: 83 d5 d1 c7
p3 = cmph.gt(r17, r21)
# CHECK: a3 d5 d1 c7
p3 = cmph.gtu(r17, r21)
# CHECK: ab c2 11 dd
p3 = cmph.eq(r17, #21)
# CHECK: ab c2 31 dd
p3 = cmph.gt(r17, #21)
# CHECK: ab c2 51 dd
p3 = cmph.gtu(r17, #21)

# Compare doublewords
# CHECK: 03 de 94 d2
p3 = cmp.eq(r21:20, r31:30)
# CHECK: 43 de 94 d2
p3 = cmp.gt(r21:20, r31:30)
# CHECK: 83 de 94 d2
p3 = cmp.gtu(r21:20, r31:30)

# Compare bitmask
# CHECK: 03 d5 91 85
p3 = bitsclr(r17, #21)
# CHECK: 03 d5 b1 85
p3 = !bitsclr(r17, #21)
# CHECK: 03 d5 51 c7
p3 = bitsset(r17, r21)
# CHECK: 03 d5 71 c7
p3 = !bitsset(r17, r21)
# CHECK: 03 d5 91 c7
p3 = bitsclr(r17, r21)
# CHECK: 03 d5 b1 c7
p3 = !bitsclr(r17, r21)

# mask generate from predicate
# CHECK: 10 c3 00 86
r17:16 = mask(p3)

# Check for TLB match
# CHECK: 63 f5 10 d2
p3 = tlbmatch(r17:16, r21)

# Predicate Transfer
# CHECK: 03 c0 45 85
p3 = r5
# CHECK: 05 c0 43 89
r5 = p3

# Test bit
# CHECK: 03 d5 11 85
p3 = tstbit(r17, #21)
# CHECK: 03 d5 31 85
p3 = !tstbit(r17, #21)
# CHECK: 03 d5 11 c7
p3 = tstbit(r17, r21)
# CHECK: 03 d5 31 c7
p3 = !tstbit(r17, r21)

# Vector compare halfwords
# CHECK: 63 de 14 d2
p3 = vcmph.eq(r21:20, r31:30)
# CHECK: 83 de 14 d2
p3 = vcmph.gt(r21:20, r31:30)
# CHECK: a3 de 14 d2
p3 = vcmph.gtu(r21:20, r31:30)
# CHECK: eb c3 14 dc
p3 = vcmph.eq(r21:20, #31)
# CHECK: eb c3 34 dc
p3 = vcmph.gt(r21:20, #31)
# CHECK: eb c3 54 dc
p3 = vcmph.gtu(r21:20, #31)

# Vector compare bytes for any match
# CHECK: 03 fe 14 d2
p3 = any8(vcmpb.eq(r21:20, r31:30))

# Vector compare bytes
# CHECK: 63 de 14 d2
p3 = vcmph.eq(r21:20, r31:30)
# CHECK: 83 de 14 d2
p3 = vcmph.gt(r21:20, r31:30)
# CHECK: a3 de 14 d2
p3 = vcmph.gtu(r21:20, r31:30)
# CHECK: eb c3 14 dc
p3 = vcmph.eq(r21:20, #31)
# CHECK: eb c3 34 dc
p3 = vcmph.gt(r21:20, #31)
# CHECK: eb c3 54 dc
p3 = vcmph.gtu(r21:20, #31)

# Vector compare words
# CHECK: 03 de 14 d2
p3 = vcmpw.eq(r21:20, r31:30)
# CHECK: 23 de 14 d2
p3 = vcmpw.gt(r21:20, r31:30)
# CHECK: 43 de 14 d2
p3 = vcmpw.gtu(r21:20, r31:30)
# CHECK: f3 c3 14 dc
p3 = vcmpw.eq(r21:20, #31)
# CHECK: f3 c3 34 dc
p3 = vcmpw.gt(r21:20, #31)
# CHECK: f3 c3 54 dc
p3 = vcmpw.gtu(r21:20, #31)

# Viterbi pack even and odd predicate bits
# CHECK: 11 c2 03 89
r17 = vitpack(p3, p2)

# Vector mux
# CHECK: 70 de 14 d1
r17:16 = vmux(p3, r21:20, r31:30)
