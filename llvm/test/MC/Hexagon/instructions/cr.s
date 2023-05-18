# RUN: llvm-mc --triple hexagon -filetype=obj -o - %s | llvm-objdump -d - | FileCheck %s
# Hexagon Programmer's Reference Manual 11.2 CR

# Corner detection acceleration
# CHECK: 93 e1 12 6b
p3 = !fastcorner9(p2, p1)
# CHECK: 91 e3 02 6b
p1 = fastcorner9(p2, p3)

# Logical reductions on predicates
# CHECK: 01 c0 82 6b
p1 = any8(p2)
# CHECK: 01 c0 a2 6b
p1 = all8(p2)

# Looping instructions
# CHECK: 00 c0 15 60
loop0(0, r21)
# CHECK: 00 c0 35 60
loop1(0, r21)
# CHECK: 60 c0 00 69
loop0(0, #12)
# CHECK: 60 c0 20 69
loop1(0, #12)

# Add to PC
# CHECK: 91 ca 49 6a
r17 = add(pc, #21)

# Pipelined loop instructions
# CHECK: 00 c0 b5 60
p3 = sp1loop0(0, r21)
# CHECK: 00 c0 d5 60
p3 = sp2loop0(0, r21)
# CHECK: 00 c0 f5 60
p3 = sp3loop0(0, r21)
# CHECK: a1 c0 a0 69
p3 = sp1loop0(0, #21)
# CHECK: a1 c0 c0 69
p3 = sp2loop0(0, #21)
# CHECK: a1 c0 e0 69
p3 = sp3loop0(0, #21)

# Logical operations on predicates
# CHECK: 01 c3 02 6b
p1 = and(p3, p2)
# CHECK: c1 c3 12 6b
p1 = and(p2, and(p3, p3))
# CHECK: 01 c3 22 6b
p1 = or(p3, p2)
# CHECK: c1 c3 32 6b
p1 = and(p2, or(p3, p3))
# CHECK: 01 c3 42 6b
p1 = xor(p2, p3)
# CHECK: c1 c3 52 6b
p1 = or(p2, and(p3, p3))
# CHECK: 01 c2 63 6b
p1 = and(p2, !p3)
# CHECK: c1 c3 72 6b
p1 = or(p2, or(p3, p3))
# CHECK: c1 c3 92 6b
p1 = and(p2, and(p3, !p3))
# CHECK: c1 c3 b2 6b
p1 = and(p2, or(p3, !p3))
# CHECK: 01 c0 c2 6b
p1 = not(p2)
# CHECK: c1 c3 d2 6b
p1 = or(p2, and(p3, !p3))
# CHECK: 01 c2 e3 6b
p1 = or(p2, !p3)
# CHECK: c1 c3 f2 6b
p1 = or(p2, or(p3, !p3))

# User control register transfer
# CHECK: 0d c0 35 62
cs1 = r21
# CHECK: 11 c0 0d 6a
r17 = cs1
