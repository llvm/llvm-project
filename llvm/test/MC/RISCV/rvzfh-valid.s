# RUN: llvm-mc %s -triple=riscv32 -mattr=+zfh -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zfh -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zfh < %s \
# RUN:     | llvm-objdump --mattr=+zfh --no-print-imm-hex -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zfh < %s \
# RUN:     | llvm-objdump --mattr=+zfh --no-print-imm-hex -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: flh ft0, 12(a0)
# CHECK-ASM: encoding: [0x07,0x10,0xc5,0x00]
flh f0, 12(a0)
# CHECK-ASM-AND-OBJ: flh ft1, 4(ra)
# CHECK-ASM: encoding: [0x87,0x90,0x40,0x00]
flh f1, +4(ra)
# CHECK-ASM-AND-OBJ: flh ft2, -2048(a3)
# CHECK-ASM: encoding: [0x07,0x91,0x06,0x80]
flh f2, -2048(x13)
# CHECK-ASM-AND-OBJ: flh ft3, -2048(s1)
# CHECK-ASM: encoding: [0x87,0x91,0x04,0x80]
flh f3, %lo(2048)(s1)
# CHECK-ASM-AND-OBJ: flh ft4, 2047(s2)
# CHECK-ASM: encoding: [0x07,0x12,0xf9,0x7f]
flh f4, 2047(s2)
# CHECK-ASM-AND-OBJ: flh ft5, 0(s3)
# CHECK-ASM: encoding: [0x87,0x92,0x09,0x00]
flh f5, 0(s3)

# CHECK-ASM-AND-OBJ: fsh ft6, 2047(s4)
# CHECK-ASM: encoding: [0xa7,0x1f,0x6a,0x7e]
fsh f6, 2047(s4)
# CHECK-ASM-AND-OBJ: fsh ft7, -2048(s5)
# CHECK-ASM: encoding: [0x27,0x90,0x7a,0x80]
fsh f7, -2048(s5)
# CHECK-ASM-AND-OBJ: fsh fs0, -2048(s6)
# CHECK-ASM: encoding: [0x27,0x10,0x8b,0x80]
fsh f8, %lo(2048)(s6)
# CHECK-ASM-AND-OBJ: fsh fs1, 999(s7)
# CHECK-ASM: encoding: [0xa7,0x93,0x9b,0x3e]
fsh f9, 999(s7)

# CHECK-ASM-AND-OBJ: fmadd.h fa0, fa1, fa2, fa3, dyn
# CHECK-ASM: encoding: [0x43,0xf5,0xc5,0x6c]
fmadd.h f10, f11, f12, f13, dyn
# CHECK-ASM-AND-OBJ: fmsub.h fa4, fa5, fa6, fa7, dyn
# CHECK-ASM: encoding: [0x47,0xf7,0x07,0x8d]
fmsub.h f14, f15, f16, f17, dyn
# CHECK-ASM-AND-OBJ: fnmsub.h fs2, fs3, fs4, fs5, dyn
# CHECK-ASM: encoding: [0x4b,0xf9,0x49,0xad]
fnmsub.h f18, f19, f20, f21, dyn
# CHECK-ASM-AND-OBJ: fnmadd.h fs6, fs7, fs8, fs9, dyn
# CHECK-ASM: encoding: [0x4f,0xfb,0x8b,0xcd]
fnmadd.h f22, f23, f24, f25, dyn

# CHECK-ASM-AND-OBJ: fadd.h fs10, fs11, ft8, dyn
# CHECK-ASM: encoding: [0x53,0xfd,0xcd,0x05]
fadd.h f26, f27, f28, dyn
# CHECK-ASM-AND-OBJ: fsub.h ft9, ft10, ft11, dyn
# CHECK-ASM: encoding: [0xd3,0x7e,0xff,0x0d]
fsub.h f29, f30, f31, dyn
# CHECK-ASM-AND-OBJ: fmul.h ft0, ft1, ft2, dyn
# CHECK-ASM: encoding: [0x53,0xf0,0x20,0x14]
fmul.h ft0, ft1, ft2, dyn
# CHECK-ASM-AND-OBJ: fdiv.h ft3, ft4, ft5, dyn
# CHECK-ASM: encoding: [0xd3,0x71,0x52,0x1c]
fdiv.h ft3, ft4, ft5, dyn
# CHECK-ASM-AND-OBJ: fsqrt.h ft6, ft7, dyn
# CHECK-ASM: encoding: [0x53,0xf3,0x03,0x5c]
fsqrt.h ft6, ft7, dyn
# CHECK-ASM-AND-OBJ: fsgnj.h fs1, fa0, fa1
# CHECK-ASM: encoding: [0xd3,0x04,0xb5,0x24]
fsgnj.h fs1, fa0, fa1
# CHECK-ASM-AND-OBJ: fsgnjn.h fa1, fa3, fa4
# CHECK-ASM: encoding: [0xd3,0x95,0xe6,0x24]
fsgnjn.h fa1, fa3, fa4
# CHECK-ASM-AND-OBJ: fsgnjx.h fa4, fa3, fa2
# CHECK-ASM: encoding: [0x53,0xa7,0xc6,0x24]
fsgnjx.h fa4, fa3, fa2
# CHECK-ASM-AND-OBJ: fmin.h fa5, fa6, fa7
# CHECK-ASM: encoding: [0xd3,0x07,0x18,0x2d]
fmin.h fa5, fa6, fa7
# CHECK-ASM-AND-OBJ: fmax.h fs2, fs3, fs4
# CHECK-ASM: encoding: [0x53,0x99,0x49,0x2d]
fmax.h fs2, fs3, fs4
# CHECK-ASM-AND-OBJ: fcvt.w.h a0, fs5, dyn
# CHECK-ASM: encoding: [0x53,0xf5,0x0a,0xc4]
fcvt.w.h a0, fs5, dyn
# CHECK-ASM-AND-OBJ: fcvt.wu.h a1, fs6, dyn
# CHECK-ASM: encoding: [0xd3,0x75,0x1b,0xc4]
fcvt.wu.h a1, fs6, dyn
# CHECK-ASM-AND-OBJ: fmv.x.h a2, fs7
# CHECK-ASM: encoding: [0x53,0x86,0x0b,0xe4]
fmv.x.h a2, fs7
# CHECK-ASM-AND-OBJ: feq.h a1, fs8, fs9
# CHECK-ASM: encoding: [0xd3,0x25,0x9c,0xa5]
feq.h a1, fs8, fs9
# CHECK-ASM-AND-OBJ: flt.h a2, fs10, fs11
# CHECK-ASM: encoding: [0x53,0x16,0xbd,0xa5]
flt.h a2, fs10, fs11
# CHECK-ASM-AND-OBJ: fle.h a3, ft8, ft9
# CHECK-ASM: encoding: [0xd3,0x06,0xde,0xa5]
fle.h a3, ft8, ft9
# CHECK-ASM-AND-OBJ: fclass.h a3, ft10
# CHECK-ASM: encoding: [0xd3,0x16,0x0f,0xe4]
fclass.h a3, ft10
# CHECK-ASM-AND-OBJ: fcvt.h.w ft11, a4, dyn
# CHECK-ASM: encoding: [0xd3,0x7f,0x07,0xd4]
fcvt.h.w ft11, a4, dyn
# CHECK-ASM-AND-OBJ: fcvt.h.wu ft0, a5, dyn
# CHECK-ASM: encoding: [0x53,0xf0,0x17,0xd4]
fcvt.h.wu ft0, a5, dyn
# CHECK-ASM-AND-OBJ: fmv.h.x ft1, a6
# CHECK-ASM: encoding: [0xd3,0x00,0x08,0xf4]
fmv.h.x ft1, a6

# Rounding modes

# CHECK-ASM-AND-OBJ: fmadd.h fa0, fa1, fa2, fa3, rne
# CHECK-ASM: encoding: [0x43,0x85,0xc5,0x6c]
fmadd.h f10, f11, f12, f13, rne
# CHECK-ASM-AND-OBJ: fmsub.h fa4, fa5, fa6, fa7, rtz
# CHECK-ASM: encoding: [0x47,0x97,0x07,0x8d]
fmsub.h f14, f15, f16, f17, rtz
# CHECK-ASM-AND-OBJ: fnmsub.h fs2, fs3, fs4, fs5, rdn
# CHECK-ASM: encoding: [0x4b,0xa9,0x49,0xad]
fnmsub.h f18, f19, f20, f21, rdn
# CHECK-ASM-AND-OBJ: fnmadd.h fs6, fs7, fs8, fs9, rup
# CHECK-ASM: encoding: [0x4f,0xbb,0x8b,0xcd]
fnmadd.h f22, f23, f24, f25, rup
# CHECK-ASM-AND-OBJ: fmadd.h fa0, fa1, fa2, fa3, rmm
# CHECK-ASM: encoding: [0x43,0xc5,0xc5,0x6c]
fmadd.h f10, f11, f12, f13, rmm
# CHECK-ASM-AND-OBJ: fmsub.h fa4, fa5, fa6, fa7
# CHECK-ASM: encoding: [0x47,0xf7,0x07,0x8d]
fmsub.h f14, f15, f16, f17, dyn

# CHECK-ASM-AND-OBJ: fadd.h fs10, fs11, ft8, rne
# CHECK-ASM: encoding: [0x53,0x8d,0xcd,0x05]
fadd.h f26, f27, f28, rne
# CHECK-ASM-AND-OBJ: fsub.h ft9, ft10, ft11, rtz
# CHECK-ASM: encoding: [0xd3,0x1e,0xff,0x0d]
fsub.h f29, f30, f31, rtz
# CHECK-ASM-AND-OBJ: fmul.h ft0, ft1, ft2, rdn
# CHECK-ASM: encoding: [0x53,0xa0,0x20,0x14]
fmul.h ft0, ft1, ft2, rdn
# CHECK-ASM-AND-OBJ: fdiv.h ft3, ft4, ft5, rup
# CHECK-ASM: encoding: [0xd3,0x31,0x52,0x1c]
fdiv.h ft3, ft4, ft5, rup

# CHECK-ASM-AND-OBJ: fsqrt.h ft6, ft7, rmm
# CHECK-ASM: encoding: [0x53,0xc3,0x03,0x5c]
fsqrt.h ft6, ft7, rmm
# CHECK-ASM-AND-OBJ: fcvt.w.h a0, fs5, rup
# CHECK-ASM: encoding: [0x53,0xb5,0x0a,0xc4]
fcvt.w.h a0, fs5, rup
# CHECK-ASM-AND-OBJ: fcvt.wu.h a1, fs6, rdn
# CHECK-ASM: encoding: [0xd3,0x25,0x1b,0xc4]
fcvt.wu.h a1, fs6, rdn
# CHECK-ASM-AND-OBJ: fcvt.h.w ft11, a4, rtz
# CHECK-ASM: encoding: [0xd3,0x1f,0x07,0xd4]
fcvt.h.w ft11, a4, rtz
# CHECK-ASM-AND-OBJ: fcvt.h.wu ft0, a5, rne
# CHECK-ASM: encoding: [0x53,0x80,0x17,0xd4]
fcvt.h.wu ft0, a5, rne
