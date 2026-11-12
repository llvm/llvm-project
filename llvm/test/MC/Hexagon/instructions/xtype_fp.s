# RUN: llvm-mc -triple=hexagon -filetype=obj -o - %s | llvm-objdump -d - | FileCheck %s
# Hexagon Programmer's Reference Manual 11.10.4 XTYPE/FP

# Floating point addition
# CHECK: 11 df 15 eb
r17 = sfadd(r21, r31)

# Classify floating-point value
# CHECK: 03 d5 f1 85
p3 = sfclass(r17, #21)
# CHECK: b3 c2 90 dc
p3 = dfclass(r17:16, #21)

# Compare floating-point value
# CHECK: 03 d5 f1 c7
p3 = sfcmp.ge(r17, r21)
# CHECK: 23 d5 f1 c7
p3 = sfcmp.uo(r17, r21)
# CHECK: 63 d5 f1 c7
p3 = sfcmp.eq(r17, r21)
# CHECK: 83 d5 f1 c7
p3 = sfcmp.gt(r17, r21)
# CHECK: 03 d4 f0 d2
p3 = dfcmp.eq(r17:16, r21:20)
# CHECK: 23 d4 f0 d2
p3 = dfcmp.gt(r17:16, r21:20)
# CHECK: 43 d4 f0 d2
p3 = dfcmp.ge(r17:16, r21:20)
# CHECK: 63 d4 f0 d2
p3 = dfcmp.uo(r17:16, r21:20)

# Convert floating-point value to other format
# CHECK: 10 c0 95 84
r17:16 = convert_sf2df(r21)
# CHECK: 31 c0 14 88
r17 = convert_df2sf(r21:20)

# Convert integer to floating-point value
# CHECK: 50 c0 f4 80
r17:16 = convert_ud2df(r21:20)
# CHECK: 70 c0 f4 80
r17:16 = convert_d2df(r21:20)
# CHECK: 30 c0 95 84
r17:16 = convert_uw2df(r21)
# CHECK: 50 c0 95 84
r17:16 = convert_w2df(r21)
# CHECK: 31 c0 34 88
r17 = convert_ud2sf(r21:20)
# CHECK: 31 c0 54 88
r17 = convert_d2sf(r21:20)
# CHECK: 11 c0 35 8b
r17 = convert_uw2sf(r21)
# CHECK: 11 c0 55 8b
r17 = convert_w2sf(r21)

# Convert floating-point value to integer
# CHECK: 10 c0 f4 80
r17:16 = convert_df2d(r21:20)
# CHECK: 30 c0 f4 80
r17:16 = convert_df2ud(r21:20)
# CHECK: d0 c0 f4 80
r17:16 = convert_df2d(r21:20):chop
# CHECK: f0 c0 f4 80
r17:16 = convert_df2ud(r21:20):chop
# CHECK: 70 c0 95 84
r17:16 = convert_sf2ud(r21)
# CHECK: 90 c0 95 84
r17:16 = convert_sf2d(r21)
# CHECK: b0 c0 95 84
r17:16 = convert_sf2ud(r21):chop
# CHECK: d0 c0 95 84
r17:16 = convert_sf2d(r21):chop
# CHECK: 31 c0 74 88
r17 = convert_df2uw(r21:20)
# CHECK: 31 c0 94 88
r17 = convert_df2w(r21:20)
# CHECK: 31 c0 b4 88
r17 = convert_df2uw(r21:20):chop
# CHECK: 31 c0 f4 88
r17 = convert_df2w(r21:20):chop
# CHECK: 11 c0 75 8b
r17 = convert_sf2uw(r21)
# CHECK: 31 c0 75 8b
r17 = convert_sf2uw(r21):chop
# CHECK: 11 c0 95 8b
r17 = convert_sf2w(r21)
# CHECK: 31 c0 95 8b
r17 = convert_sf2w(r21):chop

# Floating point extreme value assistance
# CHECK: 11 c0 b5 8b
r17 = sffixupr(r21)
# CHECK: 11 df d5 eb
r17 = sffixupn(r21, r31)
# CHECK: 31 df d5 eb
r17 = sffixupd(r21, r31)

# Floating point fused multiply-add
# CHECK: 91 df 15 ef
r17 += sfmpy(r21, r31)
# CHECK: b1 df 15 ef
r17 -= sfmpy(r21, r31)

# Floating point fused multiply-add with scaling
# CHECK: f1 df 75 ef
r17 += sfmpy(r21, r31, p3):scale

# Floating point reciprocal square root approximation
# CHECK: 71 c0 f5 8b
r17, p3 = sfinvsqrta(r21)

# Floating point fused multiply-add for library routines
# CHECK: d1 df 15 ef
r17 += sfmpy(r21, r31):lib
# CHECK: f1 df 15 ef
r17 -= sfmpy(r21, r31):lib

# Create floating-point constant
# CHECK: b1 c2 00 d6
r17 = sfmake(#21):pos
# CHECK: b1 c2 40 d6
r17 = sfmake(#21):neg
# CHECK: b0 c2 00 d9
r17:16 = dfmake(#21):pos
# CHECK: b0 c2 40 d9
r17:16 = dfmake(#21):neg

# Floating point maximum
# CHECK: 11 df 95 eb
r17 = sfmax(r21, r31)

# Floating point minimum
# CHECK: 31 df 95 eb
r17 = sfmin(r21, r31)

# Floating point multiply
# CHECK: 11 df 55 eb
r17 = sfmpy(r21, r31)

# Floating point reciprocal approximation
# CHECK: f1 df f5 eb
r17, p3 = sfrecipa(r21, r31)

# Floating point subtraction
# CHECK: 31 df 15 eb
r17 = sfsub(r21, r31)
