# RUN: llvm-mc -triple=hexagon -filetype=obj -o - %s | llvm-objdump -d - | FileCheck %s
# Hexagon Programmer's Reference Manual 11.10.5 XTYPE/MPY

# Multiply and use lower result
# CHECK: b1 df 35 d7
r17 = add(#21, mpyi(r21, r31))
# CHECK: bf d1 35 d8
r17 = add(#21, mpyi(r21, #31))
# CHECK: b5 d1 3f df
r17 = add(r21, mpyi(#84, r31))
# CHECK: f5 f1 b5 df
r17 = add(r21, mpyi(r21, #31))
# CHECK: 15 d1 1f e3
r17 = add(r21, mpyi(r17, r31))
# CHECK: f1 c3 15 e0
r17 =+ mpyi(r21, #31)
# CHECK: f1 c3 95 e0
r17 =- mpyi(r21, #31)
# CHECK: f1 c3 15 e1
r17 += mpyi(r21, #31)
# CHECK: f1 c3 95 e1
r17 -= mpyi(r21, #31)
# CHECK: 11 df 15 ed
r17 = mpyi(r21, r31)
# CHECK: 11 df 15 ef
r17 += mpyi(r21, r31)

# Vector multiply word by signed half (32x16)
# CHECK: b0 de 14 e8
r17:16 = vmpyweh(r21:20, r31:30):sat
# CHECK: b0 de 94 e8
r17:16 = vmpyweh(r21:20, r31:30):<<1:sat
# CHECK: f0 de 14 e8
r17:16 = vmpywoh(r21:20, r31:30):sat
# CHECK: f0 de 94 e8
r17:16 = vmpywoh(r21:20, r31:30):<<1:sat
# CHECK: b0 de 34 e8
r17:16 = vmpyweh(r21:20, r31:30):rnd:sat
# CHECK: b0 de b4 e8
r17:16 = vmpyweh(r21:20, r31:30):<<1:rnd:sat
# CHECK: f0 de 34 e8
r17:16 = vmpywoh(r21:20, r31:30):rnd:sat
# CHECK: f0 de b4 e8
r17:16 = vmpywoh(r21:20, r31:30):<<1:rnd:sat
# CHECK: b0 de 14 ea
r17:16 += vmpyweh(r21:20, r31:30):sat
# CHECK: b0 de 94 ea
r17:16 += vmpyweh(r21:20, r31:30):<<1:sat
# CHECK: f0 de 14 ea
r17:16 += vmpywoh(r21:20, r31:30):sat
# CHECK: f0 de 94 ea
r17:16 += vmpywoh(r21:20, r31:30):<<1:sat
# CHECK: b0 de 34 ea
r17:16 += vmpyweh(r21:20, r31:30):rnd:sat
# CHECK: b0 de b4 ea
r17:16 += vmpyweh(r21:20, r31:30):<<1:rnd:sat
# CHECK: f0 de 34 ea
r17:16 += vmpywoh(r21:20, r31:30):rnd:sat
# CHECK: f0 de b4 ea
r17:16 += vmpywoh(r21:20, r31:30):<<1:rnd:sat

# Vector multiply word by unsigned half (32x16)
# CHECK: b0 de 54 e8
r17:16 = vmpyweuh(r21:20, r31:30):sat
# CHECK: b0 de d4 e8
r17:16 = vmpyweuh(r21:20, r31:30):<<1:sat
# CHECK: f0 de 54 e8
r17:16 = vmpywouh(r21:20, r31:30):sat
# CHECK: f0 de d4 e8
r17:16 = vmpywouh(r21:20, r31:30):<<1:sat
# CHECK: b0 de 74 e8
r17:16 = vmpyweuh(r21:20, r31:30):rnd:sat
# CHECK: b0 de f4 e8
r17:16 = vmpyweuh(r21:20, r31:30):<<1:rnd:sat
# CHECK: f0 de 74 e8
r17:16 = vmpywouh(r21:20, r31:30):rnd:sat
# CHECK: f0 de f4 e8
r17:16 = vmpywouh(r21:20, r31:30):<<1:rnd:sat
# CHECK: b0 de 54 ea
r17:16 += vmpyweuh(r21:20, r31:30):sat
# CHECK: b0 de d4 ea
r17:16 += vmpyweuh(r21:20, r31:30):<<1:sat
# CHECK: f0 de 54 ea
r17:16 += vmpywouh(r21:20, r31:30):sat
# CHECK: f0 de d4 ea
r17:16 += vmpywouh(r21:20, r31:30):<<1:sat
# CHECK: b0 de 74 ea
r17:16 += vmpyweuh(r21:20, r31:30):rnd:sat
# CHECK: b0 de f4 ea
r17:16 += vmpyweuh(r21:20, r31:30):<<1:rnd:sat
# CHECK: f0 de 74 ea
r17:16 += vmpywouh(r21:20, r31:30):rnd:sat
# CHECK: f0 de f4 ea
r17:16 += vmpywouh(r21:20, r31:30):<<1:rnd:sat

# Multiply signed halfwords
# CHECK: 10 df 95 e4
r17:16 = mpy(r21.l, r31.l):<<1
# CHECK: 30 df 95 e4
r17:16 = mpy(r21.l, r31.h):<<1
# CHECK: 50 df 95 e4
r17:16 = mpy(r21.h, r31.l):<<1
# CHECK: 70 df 95 e4
r17:16 = mpy(r21.h, r31.h):<<1
# CHECK: 10 df b5 e4
r17:16 = mpy(r21.l, r31.l):<<1:rnd
# CHECK: 30 df b5 e4
r17:16 = mpy(r21.l, r31.h):<<1:rnd
# CHECK: 50 df b5 e4
r17:16 = mpy(r21.h, r31.l):<<1:rnd
# CHECK: 70 df b5 e4
r17:16 = mpy(r21.h, r31.h):<<1:rnd
# CHECK: 10 df 95 e6
r17:16 += mpy(r21.l, r31.l):<<1
# CHECK: 30 df 95 e6
r17:16 += mpy(r21.l, r31.h):<<1
# CHECK: 50 df 95 e6
r17:16 += mpy(r21.h, r31.l):<<1
# CHECK: 70 df 95 e6
r17:16 += mpy(r21.h, r31.h):<<1
# CHECK: 10 df b5 e6
r17:16 -= mpy(r21.l, r31.l):<<1
# CHECK: 30 df b5 e6
r17:16 -= mpy(r21.l, r31.h):<<1
# CHECK: 50 df b5 e6
r17:16 -= mpy(r21.h, r31.l):<<1
# CHECK: 70 df b5 e6
r17:16 -= mpy(r21.h, r31.h):<<1
# CHECK: 11 df 95 ec
r17 = mpy(r21.l, r31.l):<<1
# CHECK: 31 df 95 ec
r17 = mpy(r21.l, r31.h):<<1
# CHECK: 51 df 95 ec
r17 = mpy(r21.h, r31.l):<<1
# CHECK: 71 df 95 ec
r17 = mpy(r21.h, r31.h):<<1
# CHECK: 91 df 95 ec
r17 = mpy(r21.l, r31.l):<<1:sat
# CHECK: b1 df 95 ec
r17 = mpy(r21.l, r31.h):<<1:sat
# CHECK: d1 df 95 ec
r17 = mpy(r21.h, r31.l):<<1:sat
# CHECK: f1 df 95 ec
r17 = mpy(r21.h, r31.h):<<1:sat
# CHECK: 11 df b5 ec
r17 = mpy(r21.l, r31.l):<<1:rnd
# CHECK: 31 df b5 ec
r17 = mpy(r21.l, r31.h):<<1:rnd
# CHECK: 51 df b5 ec
r17 = mpy(r21.h, r31.l):<<1:rnd
# CHECK: 71 df b5 ec
r17 = mpy(r21.h, r31.h):<<1:rnd
# CHECK: 91 df b5 ec
r17 = mpy(r21.l, r31.l):<<1:rnd:sat
# CHECK: b1 df b5 ec
r17 = mpy(r21.l, r31.h):<<1:rnd:sat
# CHECK: d1 df b5 ec
r17 = mpy(r21.h, r31.l):<<1:rnd:sat
# CHECK: f1 df b5 ec
r17 = mpy(r21.h, r31.h):<<1:rnd:sat
# CHECK: 11 df 95 ee
r17 += mpy(r21.l, r31.l):<<1
# CHECK: 31 df 95 ee
r17 += mpy(r21.l, r31.h):<<1
# CHECK: 51 df 95 ee
r17 += mpy(r21.h, r31.l):<<1
# CHECK: 71 df 95 ee
r17 += mpy(r21.h, r31.h):<<1
# CHECK: 91 df 95 ee
r17 += mpy(r21.l, r31.l):<<1:sat
# CHECK: b1 df 95 ee
r17 += mpy(r21.l, r31.h):<<1:sat
# CHECK: d1 df 95 ee
r17 += mpy(r21.h, r31.l):<<1:sat
# CHECK: f1 df 95 ee
r17 += mpy(r21.h, r31.h):<<1:sat
# CHECK: 11 df b5 ee
r17 -= mpy(r21.l, r31.l):<<1
# CHECK: 31 df b5 ee
r17 -= mpy(r21.l, r31.h):<<1
# CHECK: 51 df b5 ee
r17 -= mpy(r21.h, r31.l):<<1
# CHECK: 71 df b5 ee
r17 -= mpy(r21.h, r31.h):<<1
# CHECK: 91 df b5 ee
r17 -= mpy(r21.l, r31.l):<<1:sat
# CHECK: b1 df b5 ee
r17 -= mpy(r21.l, r31.h):<<1:sat
# CHECK: d1 df b5 ee
r17 -= mpy(r21.h, r31.l):<<1:sat
# CHECK: f1 df b5 ee
r17 -= mpy(r21.h, r31.h):<<1:sat

# Multiply unsigned halfwords
# CHECK: 10 df d5 e4
r17:16 = mpyu(r21.l, r31.l):<<1
# CHECK: 30 df d5 e4
r17:16 = mpyu(r21.l, r31.h):<<1
# CHECK: 50 df d5 e4
r17:16 = mpyu(r21.h, r31.l):<<1
# CHECK: 70 df d5 e4
r17:16 = mpyu(r21.h, r31.h):<<1
# CHECK: 10 df d5 e6
r17:16 += mpyu(r21.l, r31.l):<<1
# CHECK: 30 df d5 e6
r17:16 += mpyu(r21.l, r31.h):<<1
# CHECK: 50 df d5 e6
r17:16 += mpyu(r21.h, r31.l):<<1
# CHECK: 70 df d5 e6
r17:16 += mpyu(r21.h, r31.h):<<1
# CHECK: 10 df f5 e6
r17:16 -= mpyu(r21.l, r31.l):<<1
# CHECK: 30 df f5 e6
r17:16 -= mpyu(r21.l, r31.h):<<1
# CHECK: 50 df f5 e6
r17:16 -= mpyu(r21.h, r31.l):<<1
# CHECK: 70 df f5 e6
r17:16 -= mpyu(r21.h, r31.h):<<1
# CHECK: 11 df d5 ec
r17 = mpyu(r21.l, r31.l):<<1
# CHECK: 31 df d5 ec
r17 = mpyu(r21.l, r31.h):<<1
# CHECK: 51 df d5 ec
r17 = mpyu(r21.h, r31.l):<<1
# CHECK: 71 df d5 ec
r17 = mpyu(r21.h, r31.h):<<1
# CHECK: 11 df d5 ee
r17 += mpyu(r21.l, r31.l):<<1
# CHECK: 31 df d5 ee
r17 += mpyu(r21.l, r31.h):<<1
# CHECK: 51 df d5 ee
r17 += mpyu(r21.h, r31.l):<<1
# CHECK: 71 df d5 ee
r17 += mpyu(r21.h, r31.h):<<1
# CHECK: 11 df f5 ee
r17 -= mpyu(r21.l, r31.l):<<1
# CHECK: 31 df f5 ee
r17 -= mpyu(r21.l, r31.h):<<1
# CHECK: 51 df f5 ee
r17 -= mpyu(r21.h, r31.l):<<1
# CHECK: 71 df f5 ee
r17 -= mpyu(r21.h, r31.h):<<1

# Polynomial multiply words
# CHECK: f0 df 55 e5
r17:16 = pmpyw(r21, r31)
# CHECK: f0 df 35 e7
r17:16 ^= pmpyw(r21, r31)

# Vector reduce multiply word by signed half (32x16)
# CHECK: 50 de 34 e8
r17:16 = vrmpywoh(r21:20, r31:30)
# CHECK: 50 de b4 e8
r17:16 = vrmpywoh(r21:20, r31:30):<<1
# CHECK: 90 de 54 e8
r17:16 = vrmpyweh(r21:20, r31:30)
# CHECK: 90 de d4 e8
r17:16 = vrmpyweh(r21:20, r31:30):<<1
# CHECK: d0 de 74 ea
r17:16 += vrmpywoh(r21:20, r31:30)
# CHECK: d0 de f4 ea
r17:16 += vrmpywoh(r21:20, r31:30):<<1
# CHECK: d0 de 34 ea
r17:16 += vrmpyweh(r21:20, r31:30)
# CHECK: d0 de b4 ea
r17:16 += vrmpyweh(r21:20, r31:30):<<1

# Multiply and use upper result
# CHECK: 31 df 15 ed
r17 = mpy(r21, r31)
# CHECK: 31 df 35 ed
r17 = mpy(r21, r31):rnd
# CHECK: 31 df 55 ed
r17 = mpyu(r21, r31)
# CHECK: 31 df 75 ed
r17 = mpysu(r21, r31)
# CHECK: 11 df b5 ed
r17 = mpy(r21, r31.h):<<1:sat
# CHECK: 31 df b5 ed
r17 = mpy(r21, r31.l):<<1:sat
# CHECK: 91 df b5 ed
r17 = mpy(r21, r31.h):<<1:rnd:sat
# CHECK: 11 df f5 ed
r17 = mpy(r21, r31):<<1:sat
# CHECK: 91 df f5 ed
r17 = mpy(r21, r31.l):<<1:rnd:sat
# CHECK: 51 df b5 ed
r17 = mpy(r21, r31):<<1
# CHECK: 11 df 75 ef
r17 += mpy(r21, r31):<<1:sat
# CHECK: 31 df 75 ef
r17 -= mpy(r21, r31):<<1:sat

# Multiply and use full result
# CHECK: 10 df 15 e5
r17:16 = mpy(r21, r31)
# CHECK: 10 df 55 e5
r17:16 = mpyu(r21, r31)
# CHECK: 10 df 15 e7
r17:16 += mpy(r21, r31)
# CHECK: 10 df 35 e7
r17:16 -= mpy(r21, r31)
# CHECK: 10 df 55 e7
r17:16 += mpyu(r21, r31)
# CHECK: 10 df 75 e7
r17:16 -= mpyu(r21, r31)

# Vector dual multiply
# CHECK: 90 de 14 e8
r17:16 = vdmpy(r21:20, r31:30):sat
# CHECK: 90 de 94 e8
r17:16 = vdmpy(r21:20, r31:30):<<1:sat
# CHECK: 90 de 14 ea
r17:16 += vdmpy(r21:20, r31:30):sat
# CHECK: 90 de 94 ea
r17:16 += vdmpy(r21:20, r31:30):<<1:sat

# Vector dual multiply with round and pack
# CHECK: 11 de 14 e9
r17 = vdmpy(r21:20, r31:30):rnd:sat
# CHECK: 11 de 94 e9
r17 = vdmpy(r21:20, r31:30):<<1:rnd:sat

# Vector reduce multiply bytes
# CHECK: 30 de 94 e8
r17:16 = vrmpybu(r21:20, r31:30)
# CHECK: 30 de d4 e8
r17:16 = vrmpybsu(r21:20, r31:30)
# CHECK: 30 de 94 ea
r17:16 += vrmpybu(r21:20, r31:30)
# CHECK: 30 de d4 ea
r17:16 += vrmpybsu(r21:20, r31:30)

# Vector dual multiply signed by unsigned bytes
# CHECK: 30 de b4 e8
r17:16 = vdmpybsu(r21:20, r31:30):sat
# CHECK: 30 de 34 ea
r17:16 += vdmpybsu(r21:20, r31:30):sat

# Vector multiply even haldwords
# CHECK: d0 de 14 e8
r17:16 = vmpyeh(r21:20, r31:30):sat
# CHECK: d0 de 94 e8
r17:16 = vmpyeh(r21:20, r31:30):<<1:sat
# CHECK: 50 de 34 ea
r17:16 += vmpyeh(r21:20, r31:30)
# CHECK: d0 de 14 ea
r17:16 += vmpyeh(r21:20, r31:30):sat
# CHECK: d0 de 94 ea
r17:16 += vmpyeh(r21:20, r31:30):<<1:sat

# Vector multiply halfwords
# CHECK: b0 df 15 e5
r17:16 = vmpyh(r21, r31):sat
# CHECK: b0 df 95 e5
r17:16 = vmpyh(r21, r31):<<1:sat
# CHECK: 30 df 35 e7
r17:16 += vmpyh(r21, r31)
# CHECK: b0 df 15 e7
r17:16 += vmpyh(r21, r31):sat
# CHECK: b0 df 95 e7
r17:16 += vmpyh(r21, r31):<<1:sat

# Vector multiply halfwords with round and pack
# CHECK: f1 df 35 ed
r17 = vmpyh(r21, r31):rnd:sat
# CHECK: f1 df b5 ed
r17 = vmpyh(r21, r31):<<1:rnd:sat

# Vector multiply halfwords signed by unsigned
# CHECK: f0 df 15 e5
r17:16 = vmpyhsu(r21, r31):sat
# CHECK: f0 df 95 e5
r17:16 = vmpyhsu(r21, r31):<<1:sat
# CHECK: b0 df 75 e7
r17:16 += vmpyhsu(r21, r31):sat
# CHECK: b0 df f5 e7
r17:16 += vmpyhsu(r21, r31):<<1:sat

# Vector reduce multiply halfwords
# CHECK: 50 de 14 e8
r17:16 = vrmpyh(r21:20, r31:30)
# CHECK: 50 de 14 ea
r17:16 += vrmpyh(r21:20, r31:30)

# Vector multiply bytes
# CHECK: 30 df 55 e5
r17:16 = vmpybsu(r21, r31)
# CHECK: 30 df 95 e5
r17:16 = vmpybu(r21, r31)
# CHECK: 30 df 95 e7
r17:16 += vmpybu(r21, r31)
# CHECK: 30 df d5 e7
r17:16 += vmpybsu(r21, r31)

# Vector polynomial multiply halfwords
# CHECK: f0 df d5 e5
r17:16 = vpmpyh(r21, r31)
# CHECK: f0 df b5 e7
r17:16 ^= vpmpyh(r21, r31)
