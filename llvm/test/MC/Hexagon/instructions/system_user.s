# RUN: llvm-mc -triple=hexagon -filetype=obj -o - %s | llvm-objdump -d - | FileCheck %s
# Hexagon Programmer's Reference Manual 11.9.1 SYSTEM/USER

# Load locked
# CHECK: 11 c0 15 92
r17 = memw_locked(r21)
# CHECK: 10 d0 15 92
r17:16 = memd_locked(r21)

# Store conditional
# CHECK: 03 d5 b1 a0
memw_locked(r17, p3) = r21
# CHECK: 03 d4 f1 a0
memd_locked(r17, p3) = r21:20

# Memory barrier
# CHECK: 00 c0 00 a8
barrier

# Data cache prefetch
# CHECK: 15 c0 11 94
dcfetch(r17 + #168)

# Send value to ETM trace
# CHECK: 00 c0 51 62
trace(r17)

# CHECK: 00 c0 00 a0
dccleana(r0)

# CHECK: 00 c0 41 a0
dccleaninva(r1)

# CHECK: 00 c0 22 a0
dcinva(r2)

# CHECK: 00 c0 c3 a0
dczeroa(r3)

# CHECK: 00 c0 c4 56
icinva(r4)

# CHECK: 02 c0 c0 57
isync

# CHECK: 00 c6 05 a6
l2fetch(r5, r6)

# CHECK: 00 c8 87 a6
l2fetch(r7, r9:8)

# CHECK: 1c df 40 54
pause(#255)

# CHECK: 00 c0 40 a8
syncht

# CHECK: 18 df 00 54
trap0(#254)

# CHECK: 14 df 80 54
trap1(#253)
