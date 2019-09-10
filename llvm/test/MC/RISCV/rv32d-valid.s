# RUN: llvm-mc %s -triple=riscv32 -mattr=+d -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+d < %s \
# RUN:     | llvm-objdump -mattr=+d -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+d -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+d < %s \
# RUN:     | llvm-objdump -mattr=+d -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ,CHECK-ASM-AND-OBJ %s

# Support for the 'D' extension implies support for 'F'
# CHECK-ASM-AND-OBJ: fadd.s fs10, fs11, ft8
# CHECK-ASM: encoding: [0x53,0xfd,0xcd,0x01]
fadd.s f26, f27, f28

# CHECK-ASM-AND-OBJ: fld ft0, 12(a0)
# CHECK-ASM: encoding: [0x07,0x30,0xc5,0x00]
fld f0, 12(a0)
# CHECK-ASM-AND-OBJ: fld ft1, 4(ra)
# CHECK-ASM: encoding: [0x87,0xb0,0x40,0x00]
fld f1, +4(ra)
# CHECK-ASM-AND-OBJ: fld ft2, -2048(a3)
# CHECK-ASM: encoding: [0x07,0xb1,0x06,0x80]
fld f2, -2048(x13)
# CHECK-ASM-AND-OBJ: fld ft3, -2048(s1)
# CHECK-ASM: encoding: [0x87,0xb1,0x04,0x80]
fld f3, %lo(2048)(s1)
# CHECK-ASM-AND-OBJ: fld ft4, 2047(s2)
# CHECK-ASM: encoding: [0x07,0x32,0xf9,0x7f]
fld f4, 2047(s2)
# CHECK-ASM-AND-OBJ: fld ft5, 0(s3)
# CHECK-ASM: encoding: [0x87,0xb2,0x09,0x00]
fld f5, 0(s3)

# CHECK-ASM-AND-OBJ: fsd ft6, 2047(s4)
# CHECK-ASM: encoding: [0xa7,0x3f,0x6a,0x7e]
fsd f6, 2047(s4)
# CHECK-ASM-AND-OBJ: fsd ft7, -2048(s5)
# CHECK-ASM: encoding: [0x27,0xb0,0x7a,0x80]
fsd f7, -2048(s5)
# CHECK-ASM-AND-OBJ: fsd fs0, -2048(s6)
# CHECK-ASM: encoding: [0x27,0x30,0x8b,0x80]
fsd f8, %lo(2048)(s6)
# CHECK-ASM-AND-OBJ: fsd fs1, 999(s7)
# CHECK-ASM: encoding: [0xa7,0xb3,0x9b,0x3e]
fsd f9, 999(s7)

# CHECK-ASM-AND-OBJ: fmadd.d fa0, fa1, fa2, fa3, dyn
# CHECK-ASM: encoding: [0x43,0xf5,0xc5,0x6a]
fmadd.d f10, f11, f12, f13, dyn
# CHECK-ASM-AND-OBJ: fmsub.d fa4, fa5, fa6, fa7, dyn
# CHECK-ASM: encoding: [0x47,0xf7,0x07,0x8b]
fmsub.d f14, f15, f16, f17, dyn
# CHECK-ASM-AND-OBJ: fnmsub.d fs2, fs3, fs4, fs5, dyn
# CHECK-ASM: encoding: [0x4b,0xf9,0x49,0xab]
fnmsub.d f18, f19, f20, f21, dyn
# CHECK-ASM-AND-OBJ: fnmadd.d fs6, fs7, fs8, fs9, dyn
# CHECK-ASM: encoding: [0x4f,0xfb,0x8b,0xcb]
fnmadd.d f22, f23, f24, f25, dyn

# CHECK-ASM-AND-OBJ: fadd.d fs10, fs11, ft8, dyn
# CHECK-ASM: encoding: [0x53,0xfd,0xcd,0x03]
fadd.d f26, f27, f28, dyn
# CHECK-ASM-AND-OBJ: fsub.d ft9, ft10, ft11, dyn
# CHECK-ASM: encoding: [0xd3,0x7e,0xff,0x0b]
fsub.d f29, f30, f31, dyn
# CHECK-ASM-AND-OBJ: fmul.d ft0, ft1, ft2, dyn
# CHECK-ASM: encoding: [0x53,0xf0,0x20,0x12]
fmul.d ft0, ft1, ft2, dyn
# CHECK-ASM-AND-OBJ: fdiv.d ft3, ft4, ft5, dyn
# CHECK-ASM: encoding: [0xd3,0x71,0x52,0x1a]
fdiv.d ft3, ft4, ft5, dyn
# CHECK-ASM-AND-OBJ: fsqrt.d ft6, ft7, dyn
# CHECK-ASM: encoding: [0x53,0xf3,0x03,0x5a]
fsqrt.d ft6, ft7, dyn
# CHECK-ASM-AND-OBJ: fsgnj.d fs1, fa0, fa1
# CHECK-ASM: encoding: [0xd3,0x04,0xb5,0x22]
fsgnj.d fs1, fa0, fa1
# CHECK-ASM-AND-OBJ: fsgnjn.d fa1, fa3, fa4
# CHECK-ASM: encoding: [0xd3,0x95,0xe6,0x22]
fsgnjn.d fa1, fa3, fa4
# CHECK-ASM-AND-OBJ: fsgnjx.d fa3, fa2, fa1
# CHECK-ASM: encoding: [0xd3,0x26,0xb6,0x22]
fsgnjx.d fa3, fa2, fa1
# CHECK-ASM-AND-OBJ: fmin.d fa5, fa6, fa7
# CHECK-ASM: encoding: [0xd3,0x07,0x18,0x2b]
fmin.d fa5, fa6, fa7
# CHECK-ASM-AND-OBJ: fmax.d fs2, fs3, fs4
# CHECK-ASM: encoding: [0x53,0x99,0x49,0x2b]
fmax.d fs2, fs3, fs4

# CHECK-ASM-AND-OBJ: fcvt.s.d fs5, fs6, dyn
# CHECK-ASM: encoding: [0xd3,0x7a,0x1b,0x40]
fcvt.s.d fs5, fs6, dyn
# CHECK-ASM-AND-OBJ: fcvt.d.s fs7, fs8
# CHECK-ASM: encoding: [0xd3,0x0b,0x0c,0x42]
fcvt.d.s fs7, fs8
# CHECK-ASM-AND-OBJ: feq.d a1, fs8, fs9
# CHECK-ASM: encoding: [0xd3,0x25,0x9c,0xa3]
feq.d a1, fs8, fs9
# CHECK-ASM-AND-OBJ: flt.d a2, fs10, fs11
# CHECK-ASM: encoding: [0x53,0x16,0xbd,0xa3]
flt.d a2, fs10, fs11
# CHECK-ASM-AND-OBJ: fle.d a3, ft8, ft9
# CHECK-ASM: encoding: [0xd3,0x06,0xde,0xa3]
fle.d a3, ft8, ft9
# CHECK-ASM-AND-OBJ: fclass.d a3, ft10
# CHECK-ASM: encoding: [0xd3,0x16,0x0f,0xe2]
fclass.d a3, ft10

# CHECK-ASM-AND-OBJ: fcvt.w.d a4, ft11, dyn
# CHECK-ASM: encoding: [0x53,0xf7,0x0f,0xc2]
fcvt.w.d a4, ft11, dyn
# CHECK-ASM-AND-OBJ: fcvt.d.w ft0, a5
# CHECK-ASM: encoding: [0x53,0x80,0x07,0xd2]
fcvt.d.w ft0, a5
# CHECK-ASM-AND-OBJ: fcvt.d.wu ft1, a6
# CHECK-ASM: encoding: [0xd3,0x00,0x18,0xd2]
fcvt.d.wu ft1, a6

# Rounding modes

# CHECK-ASM-AND-OBJ: fmadd.d fa0, fa1, fa2, fa3, rne
# CHECK-ASM: encoding: [0x43,0x85,0xc5,0x6a]
fmadd.d f10, f11, f12, f13, rne
# CHECK-ASM-AND-OBJ: fmsub.d fa4, fa5, fa6, fa7, rtz
# CHECK-ASM: encoding: [0x47,0x97,0x07,0x8b]
fmsub.d f14, f15, f16, f17, rtz
# CHECK-ASM-AND-OBJ: fnmsub.d fs2, fs3, fs4, fs5, rdn
# CHECK-ASM: encoding: [0x4b,0xa9,0x49,0xab]
fnmsub.d f18, f19, f20, f21, rdn
# CHECK-ASM-AND-OBJ: fnmadd.d fs6, fs7, fs8, fs9, rup
# CHECK-ASM: encoding: [0x4f,0xbb,0x8b,0xcb]
fnmadd.d f22, f23, f24, f25, rup

# CHECK-ASM-AND-OBJ: fadd.d fs10, fs11, ft8, rmm
# CHECK-ASM: encoding: [0x53,0xcd,0xcd,0x03]
fadd.d f26, f27, f28, rmm
# CHECK-ASM-AND-OBJ: fsub.d ft9, ft10, ft11
# CHECK-ASM: encoding: [0xd3,0x7e,0xff,0x0b]
fsub.d f29, f30, f31, dyn
# CHECK-ASM-AND-OBJ: fmul.d ft0, ft1, ft2, rne
# CHECK-ASM: encoding: [0x53,0x80,0x20,0x12]
fmul.d ft0, ft1, ft2, rne
# CHECK-ASM-AND-OBJ: fdiv.d ft3, ft4, ft5, rtz
# CHECK-ASM: encoding: [0xd3,0x11,0x52,0x1a]
fdiv.d ft3, ft4, ft5, rtz

# CHECK-ASM-AND-OBJ: fsqrt.d ft6, ft7, rdn
# CHECK-ASM: encoding: [0x53,0xa3,0x03,0x5a]
fsqrt.d ft6, ft7, rdn
# CHECK-ASM-AND-OBJ: fcvt.s.d fs5, fs6, rup
# CHECK-ASM: encoding: [0xd3,0x3a,0x1b,0x40]
fcvt.s.d fs5, fs6, rup
# CHECK-ASM-AND-OBJ: fcvt.w.d a4, ft11, rmm
# CHECK-ASM: encoding: [0x53,0xc7,0x0f,0xc2]
fcvt.w.d a4, ft11, rmm
# CHECK-ASM-AND-OBJ: fcvt.wu.d a5, ft10, dyn
# CHECK-ASM: encoding: [0xd3,0x77,0x1f,0xc2]
fcvt.wu.d a5, ft10, dyn
