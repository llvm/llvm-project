# RUN: llvm-mc %s -triple=riscv32 -mattr=+f -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+f -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+f < %s \
# RUN:     | llvm-objdump -mattr=+f -d - | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+f < %s \
# RUN:     | llvm-objdump -mattr=+f -d - | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: flw ft0, 12(a0)
# CHECK: encoding: [0x07,0x20,0xc5,0x00]
flw f0, 12(a0)
# CHECK-INST: flw ft1, 4(ra)
# CHECK: encoding: [0x87,0xa0,0x40,0x00]
flw f1, +4(ra)
# CHECK-INST: flw ft2, -2048(a3)
# CHECK: encoding: [0x07,0xa1,0x06,0x80]
flw f2, -2048(x13)
# CHECK-INST: flw ft3, -2048(s1)
# CHECK: encoding: [0x87,0xa1,0x04,0x80]
flw f3, %lo(2048)(s1)
# CHECK-INST: flw ft4, 2047(s2)
# CHECK: encoding: [0x07,0x22,0xf9,0x7f]
flw f4, 2047(s2)
# CHECK-INST: flw ft5, 0(s3)
# CHECK: encoding: [0x87,0xa2,0x09,0x00]
flw f5, 0(s3)

# CHECK-INST: fsw ft6, 2047(s4)
# CHECK: encoding: [0xa7,0x2f,0x6a,0x7e]
fsw f6, 2047(s4)
# CHECK-INST: fsw ft7, -2048(s5)
# CHECK: encoding: [0x27,0xa0,0x7a,0x80]
fsw f7, -2048(s5)
# CHECK-INST: fsw fs0, -2048(s6)
# CHECK: encoding: [0x27,0x20,0x8b,0x80]
fsw f8, %lo(2048)(s6)
# CHECK-INST: fsw fs1, 999(s7)
# CHECK: encoding: [0xa7,0xa3,0x9b,0x3e]
fsw f9, 999(s7)

# CHECK-INST: fmadd.s fa0, fa1, fa2, fa3
# CHECK: encoding: [0x43,0xf5,0xc5,0x68]
fmadd.s f10, f11, f12, f13
# CHECK-INST: fmsub.s fa4, fa5, fa6, fa7
# CHECK: encoding: [0x47,0xf7,0x07,0x89]
fmsub.s f14, f15, f16, f17
# CHECK-INST: fnmsub.s fs2, fs3, fs4, fs5
# CHECK: encoding: [0x4b,0xf9,0x49,0xa9]
fnmsub.s f18, f19, f20, f21
# CHECK-INST: fnmadd.s fs6, fs7, fs8, fs9
# CHECK: encoding: [0x4f,0xfb,0x8b,0xc9]
fnmadd.s f22, f23, f24, f25

# CHECK-INST: fadd.s fs10, fs11, ft8
# CHECK: encoding: [0x53,0xfd,0xcd,0x01]
fadd.s f26, f27, f28
# CHECK-INST: fsub.s ft9, ft10, ft11
# CHECK: encoding: [0xd3,0x7e,0xff,0x09]
fsub.s f29, f30, f31
# CHECK-INST: fmul.s ft0, ft1, ft2
# CHECK: encoding: [0x53,0xf0,0x20,0x10]
fmul.s ft0, ft1, ft2
# CHECK-INST: fdiv.s ft3, ft4, ft5
# CHECK: encoding: [0xd3,0x71,0x52,0x18]
fdiv.s ft3, ft4, ft5
# CHECK-INST: fsqrt.s ft6, ft7
# CHECK: encoding: [0x53,0xf3,0x03,0x58]
fsqrt.s ft6, ft7
# CHECK-INST: fsgnj.s fs1, fa0, fa1
# CHECK: encoding: [0xd3,0x04,0xb5,0x20]
fsgnj.s fs1, fa0, fa1
# CHECK-INST: fsgnjn.s fa1, fa3, fa4
# CHECK: encoding: [0xd3,0x95,0xe6,0x20]
fsgnjn.s fa1, fa3, fa4
# CHECK-INST: fsgnjx.s fa4, fa3, fa2
# CHECK: encoding: [0x53,0xa7,0xc6,0x20]
fsgnjx.s fa4, fa3, fa2
# CHECK-INST: fmin.s fa5, fa6, fa7
# CHECK: encoding: [0xd3,0x07,0x18,0x29]
fmin.s fa5, fa6, fa7
# CHECK-INST: fmax.s fs2, fs3, fs4
# CHECK: encoding: [0x53,0x99,0x49,0x29]
fmax.s fs2, fs3, fs4
# CHECK-INST: fcvt.w.s a0, fs5
# CHECK: encoding: [0x53,0xf5,0x0a,0xc0]
fcvt.w.s a0, fs5
# CHECK-INST: fcvt.wu.s a1, fs6
# CHECK: encoding: [0xd3,0x75,0x1b,0xc0]
fcvt.wu.s a1, fs6
# CHECK-INST: fmv.x.w a2, fs7
# CHECK: encoding: [0x53,0x86,0x0b,0xe0]
fmv.x.w a2, fs7
# CHECK-INST: feq.s a1, fs8, fs9
# CHECK: encoding: [0xd3,0x25,0x9c,0xa1]
feq.s a1, fs8, fs9
# CHECK-INST: flt.s a2, fs10, fs11
# CHECK: encoding: [0x53,0x16,0xbd,0xa1]
flt.s a2, fs10, fs11
# CHECK-INST: fle.s a3, ft8, ft9
# CHECK: encoding: [0xd3,0x06,0xde,0xa1]
fle.s a3, ft8, ft9
# CHECK-INST: fclass.s a3, ft10
# CHECK: encoding: [0xd3,0x16,0x0f,0xe0]
fclass.s a3, ft10
# CHECK-INST: fcvt.s.w ft11, a4
# CHECK: encoding: [0xd3,0x7f,0x07,0xd0]
fcvt.s.w ft11, a4
# CHECK-INST: fcvt.s.wu ft0, a5
# CHECK: encoding: [0x53,0xf0,0x17,0xd0]
fcvt.s.wu ft0, a5
# CHECK-INST: fmv.w.x ft1, a6
# CHECK: encoding: [0xd3,0x00,0x08,0xf0]
fmv.w.x ft1, a6

# Rounding modes

# CHECK-INST: fmadd.s fa0, fa1, fa2, fa3, rne
# CHECK: encoding: [0x43,0x85,0xc5,0x68]
fmadd.s f10, f11, f12, f13, rne
# CHECK-INST: fmsub.s fa4, fa5, fa6, fa7, rtz
# CHECK: encoding: [0x47,0x97,0x07,0x89]
fmsub.s f14, f15, f16, f17, rtz
# CHECK-INST: fnmsub.s fs2, fs3, fs4, fs5, rdn
# CHECK: encoding: [0x4b,0xa9,0x49,0xa9]
fnmsub.s f18, f19, f20, f21, rdn
# CHECK-INST: fnmadd.s fs6, fs7, fs8, fs9, rup
# CHECK: encoding: [0x4f,0xbb,0x8b,0xc9]
fnmadd.s f22, f23, f24, f25, rup
# CHECK-INST: fmadd.s fa0, fa1, fa2, fa3, rmm
# CHECK: encoding: [0x43,0xc5,0xc5,0x68]
fmadd.s f10, f11, f12, f13, rmm
# CHECK-INST: fmsub.s fa4, fa5, fa6, fa7
# CHECK: encoding: [0x47,0xf7,0x07,0x89]
fmsub.s f14, f15, f16, f17, dyn

# CHECK-INST: fadd.s fs10, fs11, ft8, rne
# CHECK: encoding: [0x53,0x8d,0xcd,0x01]
fadd.s f26, f27, f28, rne
# CHECK-INST: fsub.s ft9, ft10, ft11, rtz
# CHECK: encoding: [0xd3,0x1e,0xff,0x09]
fsub.s f29, f30, f31, rtz
# CHECK-INST: fmul.s ft0, ft1, ft2, rdn
# CHECK: encoding: [0x53,0xa0,0x20,0x10]
fmul.s ft0, ft1, ft2, rdn
# CHECK-INST: fdiv.s ft3, ft4, ft5, rup
# CHECK: encoding: [0xd3,0x31,0x52,0x18]
fdiv.s ft3, ft4, ft5, rup

# CHECK-INST: fsqrt.s ft6, ft7, rmm
# CHECK: encoding: [0x53,0xc3,0x03,0x58]
fsqrt.s ft6, ft7, rmm
# CHECK-INST: fcvt.w.s a0, fs5, rup
# CHECK: encoding: [0x53,0xb5,0x0a,0xc0]
fcvt.w.s a0, fs5, rup
# CHECK-INST: fcvt.wu.s a1, fs6, rdn
# CHECK: encoding: [0xd3,0x25,0x1b,0xc0]
fcvt.wu.s a1, fs6, rdn
# CHECK-INST: fcvt.s.w ft11, a4, rtz
# CHECK: encoding: [0xd3,0x1f,0x07,0xd0]
fcvt.s.w ft11, a4, rtz
# CHECK-INST: fcvt.s.wu ft0, a5, rne
# CHECK: encoding: [0x53,0x80,0x17,0xd0]
fcvt.s.wu ft0, a5, rne
