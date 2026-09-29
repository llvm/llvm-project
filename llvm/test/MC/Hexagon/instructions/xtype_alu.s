# RUN: llvm-mc -triple=hexagon -filetype=obj -o - %s | llvm-objdump -d - | FileCheck %s
# Hexagon Programmer's Reference Manual 11.10.1 XTYPE/ALU

# Absolute value doubleword
# CHECK: d0 c0 94 80
r17:16 = abs(r21:20)
# CHECK: 91 c0 95 8c
r17 = abs(r21)
# CHECK: b1 c0 95 8c
r17 = abs(r21):sat

# Add and accumulate
# CHECK: ff d1 35 db
r17 = add(r21, add(r31, #23))
# CHECK: ff d1 b5 db
r17 = add(r21, sub(#23, r31))
# CHECK: f1 c2 15 e2
r17 += add(r21, #23)
# CHECK: f1 c2 95 e2
r17 -= add(r21, #23)
# CHECK: 31 df 15 ef
r17 += add(r21, r31)
# CHECK: 31 df 95 ef
r17 -= add(r21, r31)

# Add doublewords
# CHECK: f0 de 14 d3
r17:16 = add(r21:20, r31:30)
# CHECK: b0 de 74 d3
r17:16 = add(r21:20, r31:30):sat
# CHECK: d0 de 74 d3
r17:16 = add(r21:20, r31:30):raw:lo
# CHECK: f0 de 74 d3
r17:16 = add(r21:20, r31:30):raw:hi

# Add halfword
# CHECK: 11 d5 1f d5
r17 = add(r21.l, r31.l)
# CHECK: 51 d5 1f d5
r17 = add(r21.l, r31.h)
# CHECK: 91 d5 1f d5
r17 = add(r21.l, r31.l):sat
# CHECK: d1 d5 1f d5
r17 = add(r21.l, r31.h):sat
# CHECK: 11 d5 5f d5
r17 = add(r21.l, r31.l):<<16
# CHECK: 31 d5 5f d5
r17 = add(r21.l, r31.h):<<16
# CHECK: 51 d5 5f d5
r17 = add(r21.h, r31.l):<<16
# CHECK: 71 d5 5f d5
r17 = add(r21.h, r31.h):<<16
# CHECK: 91 d5 5f d5
r17 = add(r21.l, r31.l):sat:<<16
# CHECK: b1 d5 5f d5
r17 = add(r21.l, r31.h):sat:<<16
# CHECK: d1 d5 5f d5
r17 = add(r21.h, r31.l):sat:<<16
# CHECK: f1 d5 5f d5
r17 = add(r21.h, r31.h):sat:<<16

# Add or subtract doublewords with carry
# CHECK: 70 de d4 c2
r17:16 = add(r21:20, r31:30, p3):carry
# CHECK: 70 de f4 c2
r17:16 = sub(r21:20, r31:30, p3):carry

# Logical doublewords
# CHECK: 90 c0 94 80
r17:16 = not(r21:20)
# CHECK: 10 de f4 d3
r17:16 = and(r21:20, r31:30)
# CHECK: 30 d4 fe d3
r17:16 = and(r21:20, ~r31:30)
# CHECK: 50 de f4 d3
r17:16 = or(r21:20, r31:30)
# CHECK: 70 d4 fe d3
r17:16 = or(r21:20, ~r31:30)
# CHECK: 90 de f4 d3
r17:16 = xor(r21:20, r31:30)

# Logical-logical doublewords
# CHECK: 10 de 94 ca
r17:16 ^= xor(r21:20, r31:30)

# Logical-logical words
# CHECK: f1 c3 15 da
r17 |= and(r21, #31)
# CHECK: f5 c3 51 da
r17 = or(r21, and(r17, #31))
# CHECK: f1 c3 95 da
r17 |= or(r21, #31)
# CHECK: 11 df 35 ef
r17 |= and(r21, ~r31)
# CHECK: 31 df 35 ef
r17 &= and(r21, ~r31)
# CHECK: 51 df 35 ef
r17 ^= and(r21, ~r31)
# CHECK: 11 df 55 ef
r17 &= and(r21, r31)
# CHECK: 31 df 55 ef
r17 &= or(r21, r31)
# CHECK: 51 df 55 ef
r17 &= xor(r21, r31)
# CHECK: 71 df 55 ef
r17 |= and(r21, r31)
# CHECK: 71 df 95 ef
r17 ^= xor(r21, r31)
# CHECK: 11 df d5 ef
r17 |= or(r21, r31)
# CHECK: 31 df d5 ef
r17 |= xor(r21, r31)
# CHECK: 51 df d5 ef
r17 ^= and(r21, r31)
# CHECK: 71 df d5 ef
r17 ^= or(r21, r31)

# Maximum words
# CHECK: 11 df d5 d5
r17 = max(r21, r31)
# CHECK: 91 df d5 d5
r17 = maxu(r21, r31)

# Maximum doublewords
# CHECK: 90 de d4 d3
r17:16 = max(r21:20, r31:30)
# CHECK: b0 de d4 d3
r17:16 = maxu(r21:20, r31:30)

# Minimum words
# CHECK: 11 d5 bf d5
r17 = min(r21, r31)
# CHECK: 91 d5 bf d5
r17 = minu(r21, r31)

# Minimum doublewords
# CHECK: d0 d4 be d3
r17:16 = min(r21:20, r31:30)
# CHECK: f0 d4 be d3
r17:16 = minu(r21:20, r31:30)

# Module wrap
# CHECK: f1 df f5 d3
r17 = modwrap(r21, r31)

# Negate
# CHECK: b0 c0 94 80
r17:16 = neg(r21:20)
# CHECK: d1 c0 95 8c
r17 = neg(r21):sat

# Round
# CHECK: 31 c0 d4 88
r17 = round(r21:20):sat
# CHECK: 11 df f5 8c
r17 = cround(r21, #31)
# CHECK: 91 df f5 8c
r17 = round(r21, #31)
# CHECK: d1 df f5 8c
r17 = round(r21, #31):sat
# CHECK: 11 df d5 c6
r17 = cround(r21, r31)
# CHECK: 91 df d5 c6
r17 = round(r21, r31)
# CHECK: d1 df d5 c6
r17 = round(r21, r31):sat

# Subtract doublewords
# CHECK: f0 d4 3e d3
r17:16 = sub(r21:20, r31:30)

# Subtract and accumulate words
# CHECK: 71 d5 1f ef
r17 += sub(r21, r31)

# Subtract halfword
# CHECK: 11 d5 3f d5
r17 = sub(r21.l, r31.l)
# CHECK: 51 d5 3f d5
r17 = sub(r21.l, r31.h)
# CHECK: 91 d5 3f d5
r17 = sub(r21.l, r31.l):sat
# CHECK: d1 d5 3f d5
r17 = sub(r21.l, r31.h):sat
# CHECK: 11 d5 7f d5
r17 = sub(r21.l, r31.l):<<16
# CHECK: 31 d5 7f d5
r17 = sub(r21.l, r31.h):<<16
# CHECK: 51 d5 7f d5
r17 = sub(r21.h, r31.l):<<16
# CHECK: 71 d5 7f d5
r17 = sub(r21.h, r31.h):<<16
# CHECK: 91 d5 7f d5
r17 = sub(r21.l, r31.l):sat:<<16
# CHECK: b1 d5 7f d5
r17 = sub(r21.l, r31.h):sat:<<16
# CHECK: d1 d5 7f d5
r17 = sub(r21.h, r31.l):sat:<<16
# CHECK: f1 d5 7f d5
r17 = sub(r21.h, r31.h):sat:<<16

# Sign extend word to doubleword
# CHECK: 10 c0 55 84
r17:16 = sxtw(r21)

# Vector absolute value halfwords
# CHECK: 90 c0 54 80
r17:16 = vabsh(r21:20)
# CHECK: b0 c0 54 80
r17:16 = vabsh(r21:20):sat

# Vector absolute value words
# CHECK: d0 c0 54 80
r17:16 = vabsw(r21:20)
# CHECK: f0 c0 54 80
r17:16 = vabsw(r21:20):sat

# Vector absolute difference halfwords
# CHECK: 10 d4 7e e8
r17:16 = vabsdiffh(r21:20, r31:30)

# Vector absolute difference words
# CHECK: 10 d4 3e e8
r17:16 = vabsdiffw(r21:20, r31:30)

# Vector add halfwords
# CHECK: 50 de 14 d3
r17:16 = vaddh(r21:20, r31:30)
# CHECK: 70 de 14 d3
r17:16 = vaddh(r21:20, r31:30):sat
# CHECK: 90 de 14 d3
r17:16 = vadduh(r21:20, r31:30):sat

# Vector add halfwords with saturate and pack to unsigned bytes
# CHECK: 31 de 54 c1
r17 = vaddhub(r21:20, r31:30):sat

# Vector reduce add unsigned bytes
# CHECK: 30 de 54 e8
r17:16 = vraddub(r21:20, r31:30)
# CHECK: 30 de 54 ea
r17:16 += vraddub(r21:20, r31:30)

# Vector reduce add halfwords
# CHECK: 31 de 14 e9
r17 = vradduh(r21:20, r31:30)
# CHECK: f1 de 34 e9
r17 = vraddh(r21:20, r31:30)

# Vector add bytes
# CHECK: 10 de 14 d3
r17:16 = vaddub(r21:20, r31:30)
# CHECK: 30 de 14 d3
r17:16 = vaddub(r21:20, r31:30):sat

# Vector add words
# CHECK: b0 de 14 d3
r17:16 = vaddw(r21:20, r31:30)
# CHECK: d0 de 14 d3
r17:16 = vaddw(r21:20, r31:30):sat

# Vector average halfwords
# CHECK: 50 de 54 d3
r17:16 = vavgh(r21:20, r31:30)
# CHECK: 70 de 54 d3
r17:16 = vavgh(r21:20, r31:30):rnd
# CHECK: 90 de 54 d3
r17:16 = vavgh(r21:20, r31:30):crnd
# CHECK: b0 de 54 d3
r17:16 = vavguh(r21:20, r31:30)
# CHECK: d0 de 54 d3
r17:16 = vavguh(r21:20, r31:30):rnd
# CHECK: 10 d4 9e d3
r17:16 = vnavgh(r21:20, r31:30)
# CHECK: 30 d4 9e d3
r17:16 = vnavgh(r21:20, r31:30):rnd:sat
# CHECK: 50 d4 9e d3
r17:16 = vnavgh(r21:20, r31:30):crnd:sat

# Vector average unsigned bytes
# CHECK: 10 de 54 d3
r17:16 = vavgub(r21:20, r31:30)
# CHECK: 30 de 54 d3
r17:16 = vavgub(r21:20, r31:30):rnd

# Vector average words
# CHECK: 10 de 74 d3
r17:16 = vavgw(r21:20, r31:30)
# CHECK: 30 de 74 d3
r17:16 = vavgw(r21:20, r31:30):rnd
# CHECK: 50 de 74 d3
r17:16 = vavgw(r21:20, r31:30):crnd
# CHECK: 70 de 74 d3
r17:16 = vavguw(r21:20, r31:30)
# CHECK: 90 de 74 d3
r17:16 = vavguw(r21:20, r31:30):rnd
# CHECK: 70 d4 9e d3
r17:16 = vnavgw(r21:20, r31:30)
# CHECK: 90 d4 9e d3
r17:16 = vnavgw(r21:20, r31:30):rnd:sat
# CHECK: d0 d4 9e d3
r17:16 = vnavgw(r21:20, r31:30):crnd:sat

# Vector conditional negate
# CHECK: 50 df d4 c3
r17:16 = vcnegh(r21:20, r31)

# CHECK: f0 ff 34 cb
r17:16 += vrcnegh(r21:20, r31)

# Vector maximum bytes
# CHECK: 10 d4 de d3
r17:16 = vmaxub(r21:20, r31:30)
# CHECK: d0 d4 de d3
r17:16 = vmaxb(r21:20, r31:30)

# Vector maximum halfwords
# CHECK: 30 d4 de d3
r17:16 = vmaxh(r21:20, r31:30)
# CHECK: 50 d4 de d3
r17:16 = vmaxuh(r21:20, r31:30)

# Vector reduce maximum halfwords
# CHECK: 3f d0 34 cb
r17:16 = vrmaxh(r21:20, r31)
# CHECK: 3f f0 34 cb
r17:16 = vrmaxuh(r21:20, r31)

# Vector reduce maximum words
# CHECK: 5f d0 34 cb
r17:16 = vrmaxw(r21:20, r31)
# CHECK: 5f f0 34 cb
r17:16 = vrmaxuw(r21:20, r31)

# Vector maximum words
# CHECK: b0 d4 be d3
r17:16 = vmaxuw(r21:20, r31:30)
# CHECK: 70 d4 de d3
r17:16 = vmaxw(r21:20, r31:30)

# Vector minimum bytes
# CHECK: 10 d4 be d3
r17:16 = vminub(r21:20, r31:30)
# CHECK: f0 d4 de d3
r17:16 = vminb(r21:20, r31:30)

# Vector minimum halfwords
# CHECK: 30 d4 be d3
r17:16 = vminh(r21:20, r31:30)
# CHECK: 50 d4 be d3
r17:16 = vminuh(r21:20, r31:30)

# Vector reduce minimum halfwords
# CHECK: bf d0 34 cb
r17:16 = vrminh(r21:20, r31)
# CHECK: bf f0 34 cb
r17:16 = vrminuh(r21:20, r31)

# Vector reduce minimum words
# CHECK: df d0 34 cb
r17:16 = vrminw(r21:20, r31)
# CHECK: df f0 34 cb
r17:16 = vrminuw(r21:20, r31)

# Vector minimum words
# CHECK: 70 d4 be d3
r17:16 = vminw(r21:20, r31:30)
# CHECK: 90 d4 be d3
r17:16 = vminuw(r21:20, r31:30)

# Vector sum of absolute differences unsigned bytes
# CHECK: 50 de 54 e8
r17:16 = vrsadub(r21:20, r31:30)
# CHECK: 50 de 54 ea
r17:16 += vrsadub(r21:20, r31:30)

# Vector subtract halfwords
# CHECK: 50 d4 3e d3
r17:16 = vsubh(r21:20, r31:30)
# CHECK: 70 d4 3e d3
r17:16 = vsubh(r21:20, r31:30):sat
# CHECK: 90 d4 3e d3
r17:16 = vsubuh(r21:20, r31:30):sat

# Vector subtract bytes
# CHECK: 10 d4 3e d3
r17:16 = vsubub(r21:20, r31:30)
# CHECK: 30 d4 3e d3
r17:16 = vsubub(r21:20, r31:30):sat

# Vector subtract words
# CHECK: b0 d4 3e d3
r17:16 = vsubw(r21:20, r31:30)
# CHECK: d0 d4 3e d3
r17:16 = vsubw(r21:20, r31:30):sat
