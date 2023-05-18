# RUN: llvm-mc -triple=hexagon -filetype=obj -o - %s | llvm-objdump -d - | FileCheck %s
# Hexagon Programmer's Reference Manual 11.10.3 XTYPE/COMPLEX

# Complex add/sub halfwords
# CHECK: 90 de 54 c1
r17:16 = vxaddsubh(r21:20, r31:30):sat
# CHECK: d0 de 54 c1
r17:16 = vxsubaddh(r21:20, r31:30):sat
# CHECK: 10 de d4 c1
r17:16 = vxaddsubh(r21:20, r31:30):rnd:>>1:sat
# CHECK: 50 de d4 c1
r17:16 = vxsubaddh(r21:20, r31:30):rnd:>>1:sat

# Complex add/sub words
# CHECK: 10 de 54 c1
r17:16 = vxaddsubw(r21:20, r31:30):sat
# CHECK: 50 de 54 c1
r17:16 = vxsubaddw(r21:20, r31:30):sat

# Complex multiply
# CHECK: d0 df 15 e5
r17:16 = cmpy(r21, r31):sat
# CHECK: d0 df 95 e5
r17:16 = cmpy(r21, r31):<<1:sat
# CHECK: d0 df 55 e5
r17:16 = cmpy(r21, r31*):sat
# CHECK: d0 df d5 e5
r17:16 = cmpy(r21, r31*):<<1:sat
# CHECK: d0 df 15 e7
r17:16 += cmpy(r21, r31):sat
# CHECK: d0 df 95 e7
r17:16 += cmpy(r21, r31):<<1:sat
# CHECK: f0 df 15 e7
r17:16 -= cmpy(r21, r31):sat
# CHECK: f0 df 95 e7
r17:16 -= cmpy(r21, r31):<<1:sat
# CHECK: d0 df 55 e7
r17:16 += cmpy(r21, r31*):sat
# CHECK: d0 df d5 e7
r17:16 += cmpy(r21, r31*):<<1:sat
# CHECK: f0 df 55 e7
r17:16 -= cmpy(r21, r31*):sat
# CHECK: f0 df d5 e7
r17:16 -= cmpy(r21, r31*):<<1:sat

# Complex multiply real or imaginary
# CHECK: 30 df 15 e5
r17:16 = cmpyi(r21, r31)
# CHECK: 50 df 15 e5
r17:16 = cmpyr(r21, r31)
# CHECK: 30 df 15 e7
r17:16 += cmpyi(r21, r31)
# CHECK: 50 df 15 e7
r17:16 += cmpyr(r21, r31)

# Complex multiply with round and pack
# CHECK: d1 df 35 ed
r17 = cmpy(r21, r31):rnd:sat
# CHECK: d1 df b5 ed
r17 = cmpy(r21, r31):<<1:rnd:sat
# CHECK: d1 df 75 ed
r17 = cmpy(r21, r31*):rnd:sat
# CHECK: d1 df f5 ed
r17 = cmpy(r21, r31*):<<1:rnd:sat

# Complex multiply 32x16
# CHECK: 91 df 14 c5
r17 = cmpyiwh(r21:20, r31):<<1:rnd:sat
# CHECK: b1 df 14 c5
r17 = cmpyiwh(r21:20, r31*):<<1:rnd:sat
# CHECK: d1 df 14 c5
r17 = cmpyrwh(r21:20, r31):<<1:rnd:sat
# CHECK: f1 df 14 c5
r17 = cmpyrwh(r21:20, r31*):<<1:rnd:sat

# Vector complex multiply real or imaginary
# CHECK: d0 de 34 e8
r17:16 = vcmpyr(r21:20, r31:30):sat
# CHECK: d0 de b4 e8
r17:16 = vcmpyr(r21:20, r31:30):<<1:sat
# CHECK: d0 de 54 e8
r17:16 = vcmpyi(r21:20, r31:30):sat
# CHECK: d0 de d4 e8
r17:16 = vcmpyi(r21:20, r31:30):<<1:sat
# CHECK: 90 de 34 ea
r17:16 += vcmpyr(r21:20, r31:30):sat
# CHECK: 90 de 54 ea
r17:16 += vcmpyi(r21:20, r31:30):sat

# Vector complex conjugate
# CHECK: f0 c0 94 80
r17:16 = vconj(r21:20):sat

# Vector complex rotate
# CHECK: 10 df d4 c3
r17:16 = vcrotate(r21:20, r31)

# Vector reduce complex multiply real or imaginary
# CHECK: 10 de 14 e8
r17:16 = vrcmpyi(r21:20, r31:30)
# CHECK: 30 de 14 e8
r17:16 = vrcmpyr(r21:20, r31:30)
# CHECK: 10 de 54 e8
r17:16 = vrcmpyi(r21:20, r31:30*)
# CHECK: 30 de 74 e8
r17:16 = vrcmpyr(r21:20, r31:30*)

# Vector reduce complex multiply by scalar
# CHECK: 90 de b4 e8
r17:16 = vrcmpys(r21:20, r31:30):<<1:sat:raw:hi
# CHECK: 90 de f4 e8
r17:16 = vrcmpys(r21:20, r31:30):<<1:sat:raw:lo
# CHECK: 90 de b4 ea
r17:16 += vrcmpys(r21:20, r31:30):<<1:sat:raw:hi
# CHECK: 90 de f4 ea
r17:16 += vrcmpys(r21:20, r31:30):<<1:sat:raw:lo

# Vector reduce complex multiply by scalar with round and pack
# CHECK: d1 de b4 e9
r17 = vrcmpys(r21:20, r31:30):<<1:rnd:sat:raw:hi
# CHECK: f1 de b4 e9
r17 = vrcmpys(r21:20, r31:30):<<1:rnd:sat:raw:lo

# Vector reduce complex rotate
# CHECK: f0 ff d4 c3
r17:16 = vrcrotate(r21:20, r31, #3)
# CHECK: 30 ff b4 cb
r17:16 += vrcrotate(r21:20, r31, #3)
