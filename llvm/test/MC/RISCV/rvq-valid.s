# RUN: llvm-mc %s -triple=riscv32 -mattr=+q -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+q < %s \
# RUN:     | llvm-objdump --no-print-imm-hex --mattr=+q -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+q -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+q < %s \
# RUN:     | llvm-objdump --no-print-imm-hex --mattr=+q -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# Support for the 'Q' extension implies support for 'D' and 'F'

# CHECK-ASM-AND-OBJ: fadd.d fs10, fs11, ft8, dyn
# CHECK-ASM: encoding: [0x53,0xfd,0xcd,0x03]
fadd.d f26, f27, f28, dyn

# CHECK-ASM-AND-OBJ: fadd.s fs10, fs11, ft8
# CHECK-ASM: encoding: [0x53,0xfd,0xcd,0x01]
fadd.s f26, f27, f28

# CHECK-ASM-AND-OBJ: flq ft0, 12(a0)
# CHECK-ASM: encoding: [0x07,0x40,0xc5,0x00]
flq f0, 12(a0)
# CHECK-ASM-AND-OBJ: flq ft1, 4(ra)
# CHECK-ASM: encoding: [0x87,0xc0,0x40,0x00]
flq f1, +4(ra)
# CHECK-ASM-AND-OBJ: flq ft2, -2048(a3)
# CHECK-ASM: encoding: [0x07,0xc1,0x06,0x80]
flq f2, -2048(x13)
# CHECK-ASM: flq ft3, %lo(2048)(s1)  # encoding: [0x87,0xc1,0bAAAA0100,A]
# CHECK-OBJ: flq ft3, -2048(s1)
flq f3, %lo(2048)(s1)
# CHECK-ASM-AND-OBJ: flq ft4, 2047(s2)
# CHECK-ASM: encoding: [0x07,0x42,0xf9,0x7f]
flq f4, 2047(s2)
# CHECK-ASM-AND-OBJ: flq ft5, 0(s3)
# CHECK-ASM: encoding: [0x87,0xc2,0x09,0x00]
flq f5, 0(s3)

# CHECK-ASM-AND-OBJ: fsq ft6, 2047(s4)
# CHECK-ASM: encoding: [0xa7,0x4f,0x6a,0x7e]
fsq f6, 2047(s4)
# CHECK-ASM-AND-OBJ: fsq ft7, -2048(s5)
# CHECK-ASM: encoding: [0x27,0xc0,0x7a,0x80]
fsq f7, -2048(s5)
# CHECK-ASM: fsq fs0, %lo(2048)(s6)  # encoding: [0x27'A',0x40'A',0x8b'A',A]
# CHECK-OBJ: fsq fs0, -2048(s6)
fsq f8, %lo(2048)(s6)
# CHECK-ASM-AND-OBJ: fsq fs1, 999(s7)
# CHECK-ASM: encoding: [0xa7,0xc3,0x9b,0x3e]
fsq f9, 999(s7)

# CHECK-ASM-AND-OBJ: fmadd.q fa0, fa1, fa2, fa3, dyn
# CHECK-ASM: encoding: [0x43,0xf5,0xc5,0x6e]
fmadd.q f10, f11, f12, f13, dyn
# CHECK-ASM-AND-OBJ: fmsub.q fa4, fa5, fa6, fa7, dyn
# CHECK-ASM: encoding: [0x47,0xf7,0x07,0x8f]
fmsub.q f14, f15, f16, f17, dyn
# CHECK-ASM-AND-OBJ: fnmsub.q fs2, fs3, fs4, fs5, dyn
# CHECK-ASM: encoding: [0x4b,0xf9,0x49,0xaf]
fnmsub.q f18, f19, f20, f21, dyn
# CHECK-ASM-AND-OBJ: fnmadd.q fs6, fs7, fs8, fs9, dyn
# CHECK-ASM: encoding: [0x4f,0xfb,0x8b,0xcf]
fnmadd.q f22, f23, f24, f25, dyn

# CHECK-ASM-AND-OBJ: fadd.q fs10, fs11, ft8, dyn
# CHECK-ASM: encoding: [0x53,0xfd,0xcd,0x07]
fadd.q f26, f27, f28, dyn
# CHECK-ASM-AND-OBJ: fsub.q ft9, ft10, ft11, dyn
# CHECK-ASM: encoding: [0xd3,0x7e,0xff,0x0f]
fsub.q f29, f30, f31, dyn
# CHECK-ASM-AND-OBJ: fmul.q ft0, ft1, ft2, dyn
# CHECK-ASM: encoding: [0x53,0xf0,0x20,0x16]
fmul.q ft0, ft1, ft2, dyn
# CHECK-ASM-AND-OBJ: fdiv.q ft3, ft4, ft5, dyn
# CHECK-ASM: encoding: [0xd3,0x71,0x52,0x1e]
fdiv.q ft3, ft4, ft5, dyn
# CHECK-ASM-AND-OBJ: fsqrt.q ft6, ft7, dyn
# CHECK-ASM: encoding: [0x53,0xf3,0x03,0x5e]
fsqrt.q ft6, ft7, dyn
# CHECK-ASM-AND-OBJ: fsgnj.q fs1, fa0, fa1
# CHECK-ASM: encoding: [0xd3,0x04,0xb5,0x26]
fsgnj.q fs1, fa0, fa1
# CHECK-ASM-AND-OBJ: fsgnjn.q fa1, fa3, fa4
# CHECK-ASM: encoding: [0xd3,0x95,0xe6,0x26]
fsgnjn.q fa1, fa3, fa4
# CHECK-ASM-AND-OBJ: fsgnjx.q fa3, fa2, fa1
# CHECK-ASM: encoding: [0xd3,0x26,0xb6,0x26]
fsgnjx.q fa3, fa2, fa1
# CHECK-ASM-AND-OBJ: fmin.q fa5, fa6, fa7
# CHECK-ASM: encoding: [0xd3,0x07,0x18,0x2f]
fmin.q fa5, fa6, fa7
# CHECK-ASM-AND-OBJ: fmax.q fs2, fs3, fs4
# CHECK-ASM: encoding: [0x53,0x99,0x49,0x2f]
fmax.q fs2, fs3, fs4

# CHECK-ASM-AND-OBJ: fcvt.s.q fs5, fs6, dyn
# CHECK-ASM: encoding: [0xd3,0x7a,0x3b,0x40]
fcvt.s.q fs5, fs6, dyn
# CHECK-ASM-AND-OBJ: fcvt.q.s fs7, fs8
# CHECK-ASM: encoding: [0xd3,0x0b,0x0c,0x46]
fcvt.q.s fs7, fs8
# CHECK-ASM-AND-OBJ: fcvt.q.s fs7, fs8, rup
# CHECK-ASM: encoding: [0xd3,0x3b,0x0c,0x46]
fcvt.q.s fs7, fs8, rup
# CHECK-ASM-AND-OBJ: fcvt.d.q fs5, fs6, dyn
# CHECK-ASM: encoding: [0xd3,0x7a,0x3b,0x42]
fcvt.d.q fs5, fs6, dyn
# CHECK-ASM-AND-OBJ: fcvt.q.d fs7, fs8
# CHECK-ASM: encoding: [0xd3,0x0b,0x1c,0x46]
fcvt.q.d fs7, fs8
# CHECK-ASM-AND-OBJ: fcvt.q.d fs7, fs8, rup
# CHECK-ASM: encoding: [0xd3,0x3b,0x1c,0x46]
fcvt.q.d fs7, fs8, rup
# CHECK-ASM-AND-OBJ: feq.q a1, fs8, fs9
# CHECK-ASM: encoding: [0xd3,0x25,0x9c,0xa7]
feq.q a1, fs8, fs9
# CHECK-ASM-AND-OBJ: flt.q a2, fs10, fs11
# CHECK-ASM: encoding: [0x53,0x16,0xbd,0xa7]
flt.q a2, fs10, fs11
# CHECK-ASM-AND-OBJ: fle.q a3, ft8, ft9
# CHECK-ASM: encoding: [0xd3,0x06,0xde,0xa7]
fle.q a3, ft8, ft9
# CHECK-ASM-AND-OBJ: fclass.q a3, ft10
# CHECK-ASM: encoding: [0xd3,0x16,0x0f,0xe6]
fclass.q a3, ft10

# CHECK-ASM-AND-OBJ: fcvt.w.q a4, ft11, dyn
# CHECK-ASM: encoding: [0x53,0xf7,0x0f,0xc6]
fcvt.w.q a4, ft11, dyn
# CHECK-ASM-AND-OBJ: fcvt.q.w ft0, a5
# CHECK-ASM: encoding: [0x53,0x80,0x07,0xd6]
fcvt.q.w ft0, a5
# CHECK-ASM-AND-OBJ: fcvt.q.w ft0, a5, rup
# CHECK-ASM: encoding: [0x53,0xb0,0x07,0xd6]
fcvt.q.w ft0, a5, rup
# CHECK-ASM-AND-OBJ: fcvt.q.wu ft1, a6
# CHECK-ASM: encoding: [0xd3,0x00,0x18,0xd6]
fcvt.q.wu ft1, a6
# CHECK-ASM-AND-OBJ: fcvt.q.wu ft1, a6, rup
# CHECK-ASM: encoding: [0xd3,0x30,0x18,0xd6]
fcvt.q.wu ft1, a6, rup

# Rounding modes

# CHECK-ASM-AND-OBJ: fmadd.q fa0, fa1, fa2, fa3, rne
# CHECK-ASM: encoding: [0x43,0x85,0xc5,0x6e]
fmadd.q f10, f11, f12, f13, rne
# CHECK-ASM-AND-OBJ: fmsub.q fa4, fa5, fa6, fa7, rtz
# CHECK-ASM: encoding: [0x47,0x97,0x07,0x8f]
fmsub.q f14, f15, f16, f17, rtz
# CHECK-ASM-AND-OBJ: fnmsub.q fs2, fs3, fs4, fs5, rdn
# CHECK-ASM: encoding: [0x4b,0xa9,0x49,0xaf]
fnmsub.q f18, f19, f20, f21, rdn
# CHECK-ASM-AND-OBJ: fnmadd.q fs6, fs7, fs8, fs9, rup
# CHECK-ASM: encoding: [0x4f,0xbb,0x8b,0xcf]
fnmadd.q f22, f23, f24, f25, rup

# CHECK-ASM-AND-OBJ: fadd.q fs10, fs11, ft8, rmm
# CHECK-ASM: encoding: [0x53,0xcd,0xcd,0x07]
fadd.q f26, f27, f28, rmm
# CHECK-ASM-AND-OBJ: fsub.q ft9, ft10, ft11
# CHECK-ASM: encoding: [0xd3,0x7e,0xff,0x0f]
fsub.q f29, f30, f31, dyn
# CHECK-ASM-AND-OBJ: fmul.q ft0, ft1, ft2, rne
# CHECK-ASM: encoding: [0x53,0x80,0x20,0x16]
fmul.q ft0, ft1, ft2, rne
# CHECK-ASM-AND-OBJ: fdiv.q ft3, ft4, ft5, rtz
# CHECK-ASM: encoding: [0xd3,0x11,0x52,0x1e]
fdiv.q ft3, ft4, ft5, rtz

# CHECK-ASM-AND-OBJ: fsqrt.q ft6, ft7, rdn
# CHECK-ASM: encoding: [0x53,0xa3,0x03,0x5e]
fsqrt.q ft6, ft7, rdn
# CHECK-ASM-AND-OBJ: fcvt.s.q fs5, fs6, rup
# CHECK-ASM: encoding: [0xd3,0x3a,0x3b,0x40]
fcvt.s.q fs5, fs6, rup
# CHECK-ASM-AND-OBJ: fcvt.w.q a4, ft11, rmm
# CHECK-ASM: encoding: [0x53,0xc7,0x0f,0xc6]
fcvt.w.q a4, ft11, rmm
# CHECK-ASM-AND-OBJ: fcvt.wu.q a5, ft10, dyn
# CHECK-ASM: encoding: [0xd3,0x77,0x1f,0xc6]
fcvt.wu.q a5, ft10, dyn
