# RUN: llvm-mc -triple=hexagon -filetype=obj -o - %s | llvm-objdump -d - | FileCheck %s
# Hexagon Programmer's Reference Manual 11.10.8 XTYPE/SHIFT

# Shift by immediate
# CHECK: 10 df 14 80
r17:16 = asr(r21:20, #31)
# CHECK: 30 df 14 80
r17:16 = lsr(r21:20, #31)
# CHECK: 50 df 14 80
r17:16 = asl(r21:20, #31)
# CHECK: 11 df 15 8c
r17 = asr(r21, #31)
# CHECK: 31 df 15 8c
r17 = lsr(r21, #31)
# CHECK: 51 df 15 8c
r17 = asl(r21, #31)

# Shift by immediate and accumulate
# CHECK: 10 df 14 82
r17:16 -= asr(r21:20, #31)
# CHECK: 30 df 14 82
r17:16 -= lsr(r21:20, #31)
# CHECK: 50 df 14 82
r17:16 -= asl(r21:20, #31)
# CHECK: 90 df 14 82
r17:16 += asr(r21:20, #31)
# CHECK: b0 df 14 82
r17:16 += lsr(r21:20, #31)
# CHECK: d0 df 14 82
r17:16 += asl(r21:20, #31)
# CHECK: 11 df 15 8e
r17 -= asr(r21, #31)
# CHECK: 31 df 15 8e
r17 -= lsr(r21, #31)
# CHECK: 51 df 15 8e
r17 -= asl(r21, #31)
# CHECK: 91 df 15 8e
r17 += asr(r21, #31)
# CHECK: b1 df 15 8e
r17 += lsr(r21, #31)
# CHECK: d1 df 15 8e
r17 += asl(r21, #31)
# CHECK: 4c f7 11 de
r17 = add(#21, asl(r17, #23))
# CHECK: 4e f7 11 de
r17 = sub(#21, asl(r17, #23))
# CHECK: 5c f7 11 de
r17 = add(#21, lsr(r17, #23))
# CHECK: 5e f7 11 de
r17 = sub(#21, lsr(r17, #23))

# Shift by immediate and add
# CHECK: f1 d5 1f c4
r17 = addasl(r21, r31, #7)

# Shift by immediate and logical
# CHECK: 10 df 54 82
r17:16 &= asr(r21:20, #31)
# CHECK: 30 df 54 82
r17:16 &= lsr(r21:20, #31)
# CHECK: 50 df 54 82
r17:16 &= asl(r21:20, #31)
# CHECK: 90 df 54 82
r17:16 |= asr(r21:20, #31)
# CHECK: b0 df 54 82
r17:16 |= lsr(r21:20, #31)
# CHECK: d0 df 54 82
r17:16 |= asl(r21:20, #31)
# CHECK: 30 df 94 82
r17:16 ^= lsr(r21:20, #31)
# CHECK: 50 df 94 82
r17:16 ^= asl(r21:20, #31)
# CHECK: 11 df 55 8e
r17 &= asr(r21, #31)
# CHECK: 31 df 55 8e
r17 &= lsr(r21, #31)
# CHECK: 51 df 55 8e
r17 &= asl(r21, #31)
# CHECK: 91 df 55 8e
r17 |= asr(r21, #31)
# CHECK: b1 df 55 8e
r17 |= lsr(r21, #31)
# CHECK: d1 df 55 8e
r17 |= asl(r21, #31)
# CHECK: 31 df 95 8e
r17 ^= lsr(r21, #31)
# CHECK: 51 df 95 8e
r17 ^= asl(r21, #31)
# CHECK: 48 ff 11 de
r17 = and(#21, asl(r17, #31))
# CHECK: 4a ff 11 de
r17 = or(#21, asl(r17, #31))
# CHECK: 58 ff 11 de
r17 = and(#21, lsr(r17, #31))
# CHECK: 5a ff 11 de
r17 = or(#21, lsr(r17, #31))

# Shift right by immediate with rounding
# CHECK: f0 df d4 80
r17:16 = asr(r21:20, #31):rnd
# CHECK: 11 df 55 8c
r17 = asr(r21, #31):rnd

# Shift left by immediate with saturation
# CHECK: 51 df 55 8c
r17 = asl(r21, #31):sat

# Shift by register
# CHECK: 10 df 94 c3
r17:16 = asr(r21:20, r31)
# CHECK: 50 df 94 c3
r17:16 = lsr(r21:20, r31)
# CHECK: 90 df 94 c3
r17:16 = asl(r21:20, r31)
# CHECK: d0 df 94 c3
r17:16 = lsl(r21:20, r31)
# CHECK: 11 df 55 c6
r17 = asr(r21, r31)
# CHECK: 51 df 55 c6
r17 = lsr(r21, r31)
# CHECK: 91 df 55 c6
r17 = asl(r21, r31)
# CHECK: d1 df 55 c6
r17 = lsl(r21, r31)
# CHECK: f1 df 8a c6
r17 = lsl(#21, r31)

# Shift by register and accumulate
# CHECK: 10 df 94 cb
r17:16 -= asr(r21:20, r31)
# CHECK: 50 df 94 cb
r17:16 -= lsr(r21:20, r31)
# CHECK: 90 df 94 cb
r17:16 -= asl(r21:20, r31)
# CHECK: d0 df 94 cb
r17:16 -= lsl(r21:20, r31)
# CHECK: 10 df d4 cb
r17:16 += asr(r21:20, r31)
# CHECK: 50 df d4 cb
r17:16 += lsr(r21:20, r31)
# CHECK: 90 df d4 cb
r17:16 += asl(r21:20, r31)
# CHECK: d0 df d4 cb
r17:16 += lsl(r21:20, r31)
# CHECK: 11 df 95 cc
r17 -= asr(r21, r31)
# CHECK: 51 df 95 cc
r17 -= lsr(r21, r31)
# CHECK: 91 df 95 cc
r17 -= asl(r21, r31)
# CHECK: d1 df 95 cc
r17 -= lsl(r21, r31)
# CHECK: 11 df d5 cc
r17 += asr(r21, r31)
# CHECK: 51 df d5 cc
r17 += lsr(r21, r31)
# CHECK: 91 df d5 cc
r17 += asl(r21, r31)
# CHECK: d1 df d5 cc
r17 += lsl(r21, r31)

# Shift by register and logical
# CHECK: 10 df 14 cb
r17:16 |= asr(r21:20, r31)
# CHECK: 50 df 14 cb
r17:16 |= lsr(r21:20, r31)
# CHECK: 90 df 14 cb
r17:16 |= asl(r21:20, r31)
# CHECK: d0 df 14 cb
r17:16 |= lsl(r21:20, r31)
# CHECK: 10 df 54 cb
r17:16 &= asr(r21:20, r31)
# CHECK: 50 df 54 cb
r17:16 &= lsr(r21:20, r31)
# CHECK: 90 df 54 cb
r17:16 &= asl(r21:20, r31)
# CHECK: d0 df 54 cb
r17:16 &= lsl(r21:20, r31)
# CHECK: 10 df 74 cb
r17:16 ^= asr(r21:20, r31)
# CHECK: 50 df 74 cb
r17:16 ^= lsr(r21:20, r31)
# CHECK: 90 df 74 cb
r17:16 ^= asl(r21:20, r31)
# CHECK: d0 df 74 cb
r17:16 ^= lsl(r21:20, r31)
# CHECK: 11 df 15 cc
r17 |= asr(r21, r31)
# CHECK: 51 df 15 cc
r17 |= lsr(r21, r31)
# CHECK: 91 df 15 cc
r17 |= asl(r21, r31)
# CHECK: d1 df 15 cc
r17 |= lsl(r21, r31)
# CHECK: 11 df 55 cc
r17 &= asr(r21, r31)
# CHECK: 51 df 55 cc
r17 &= lsr(r21, r31)
# CHECK: 91 df 55 cc
r17 &= asl(r21, r31)
# CHECK: d1 df 55 cc
r17 &= lsl(r21, r31)

# Shift by register with saturation
# CHECK: 11 df 15 c6
r17 = asr(r21, r31):sat
# CHECK: 91 df 15 c6
r17 = asl(r21, r31):sat

# Vector shift halfwords by immediate
# CHECK: 10 c5 94 80
r17:16 = vasrh(r21:20, #5)
# CHECK: 30 c5 94 80
r17:16 = vlsrh(r21:20, #5)
# CHECK: 50 c5 94 80
r17:16 = vaslh(r21:20, #5)

# Vector arithmetic shift halfwords with round
# CHECK: 10 c5 34 80
r17:16 = vasrh(r21:20, #5):raw

# Vector arithmetic shift halfwords with saturate and pack
# CHECK: 91 c5 74 88
r17 = vasrhub(r21:20, #5):raw
# CHECK: b1 c5 74 88
r17 = vasrhub(r21:20, #5):sat

# Vector shift halfwords by register
# CHECK: 10 df 54 c3
r17:16 = vasrh(r21:20, r31)
# CHECK: 50 df 54 c3
r17:16 = vlsrh(r21:20, r31)
# CHECK: 90 df 54 c3
r17:16 = vaslh(r21:20, r31)
# CHECK: d0 df 54 c3
r17:16 = vlslh(r21:20, r31)

# Vector shift words by immediate
# CHECK: 10 df 54 80
r17:16 = vasrw(r21:20, #31)
# CHECK: 30 df 54 80
r17:16 = vlsrw(r21:20, #31)
# CHECK: 50 df 54 80
r17:16 = vaslw(r21:20, #31)

# Vector shift words by register
# CHECK: 10 df 14 c3
r17:16 = vasrw(r21:20, r31)
# CHECK: 50 df 14 c3
r17:16 = vlsrw(r21:20, r31)
# CHECK: 90 df 14 c3
r17:16 = vaslw(r21:20, r31)
# CHECK: d0 df 14 c3
r17:16 = vlslw(r21:20, r31)

# Vector shift words with truncate and pack
# CHECK: 51 df d4 88
r17 = vasrw(r21:20, #31)
# CHECK: 51 df 14 c5
r17 = vasrw(r21:20, r31)
