# RUN: llvm-mc %s -triple=xtensa -show-encoding --mattr=+dfpaccel \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4
LBL0:

# CHECK-INST: rur a3, f64r_lo
# CHECK: encoding: [0xa0,0x3e,0xe3]
rur a3, f64r_lo

# CHECK-INST: rur a3, f64r_lo
# CHECK: encoding: [0xa0,0x3e,0xe3]
rur a3, 234

# CHECK-INST: rur a3, f64r_lo
# CHECK: encoding: [0xa0,0x3e,0xe3]
rur.f64r_lo a3

# CHECK-INST: wur a3, f64r_lo
# CHECK: encoding: [0x30,0xea,0xf3]
wur a3, f64r_lo

# CHECK-INST: rur a3, f64r_hi
# CHECK: encoding: [0xb0,0x3e,0xe3]
rur a3, f64r_hi

# CHECK-INST: rur a3, f64r_hi
# CHECK: encoding: [0xb0,0x3e,0xe3]
rur a3, 235

# CHECK-INST: rur a3, f64r_hi
# CHECK: encoding: [0xb0,0x3e,0xe3]
rur.f64r_hi a3

# CHECK-INST: wur a3, f64r_hi
# CHECK: encoding: [0x30,0xeb,0xf3]
wur a3, f64r_hi

# CHECK-INST: rur a3, f64s
# CHECK: encoding: [0xc0,0x3e,0xe3]
rur a3, f64s

# CHECK-INST: rur a3, f64s
# CHECK: encoding: [0xc0,0x3e,0xe3]
rur a3, 236

# CHECK-INST: rur a3, f64s
# CHECK: encoding: [0xc0,0x3e,0xe3]
rur.f64s a3

# CHECK-INST: wur a3, f64s
# CHECK: encoding: [0x30,0xec,0xf3]
wur a3, f64s
