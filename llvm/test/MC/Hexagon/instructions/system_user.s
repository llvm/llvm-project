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
