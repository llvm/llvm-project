# RUN: llvm-mc %s -triple=riscv32 -mattr=+zfinx -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zfinx -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zfinx %s \
# RUN:     | llvm-objdump --mattr=+zfinx -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zfinx %s \
# RUN:     | llvm-objdump --mattr=+zfinx -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: fmadd.s a0, a1, a2, a3, dyn
# CHECK-ASM: encoding: [0x43,0xf5,0xc5,0x68]
fmadd.s x10, x11, x12, x13, dyn
# CHECK-ASM-AND-OBJ: fmsub.s a4, a5, a6, a7, dyn
# CHECK-ASM: encoding: [0x47,0xf7,0x07,0x89]
fmsub.s x14, x15, x16, x17, dyn
# CHECK-ASM-AND-OBJ: fnmsub.s s2, s3, s4, s5, dyn
# CHECK-ASM: encoding: [0x4b,0xf9,0x49,0xa9]
fnmsub.s x18, x19, x20, x21, dyn
# CHECK-ASM-AND-OBJ: fnmadd.s s6, s7, s8, s9, dyn
# CHECK-ASM: encoding: [0x4f,0xfb,0x8b,0xc9]
fnmadd.s x22, x23, x24, x25, dyn

# CHECK-ASM-AND-OBJ: fadd.s s10, s11, t3, dyn
# CHECK-ASM: encoding: [0x53,0xfd,0xcd,0x01]
fadd.s x26, x27, x28, dyn
# CHECK-ASM-AND-OBJ: fsub.s t4, t5, t6, dyn
# CHECK-ASM: encoding: [0xd3,0x7e,0xff,0x09]
fsub.s x29, x30, x31, dyn
# CHECK-ASM-AND-OBJ: fmul.s s0, s1, s2, dyn
# CHECK-ASM: encoding: [0x53,0xf4,0x24,0x11]
fmul.s s0, s1, s2, dyn
# CHECK-ASM-AND-OBJ: fdiv.s s3, s4, s5, dyn
# CHECK-ASM: encoding: [0xd3,0x79,0x5a,0x19]
fdiv.s s3, s4, s5, dyn
# CHECK-ASM-AND-OBJ: fsqrt.s t1, t2, dyn
# CHECK-ASM: encoding: [0x53,0xf3,0x03,0x58]
fsqrt.s t1, t2, dyn
# CHECK-ASM-AND-OBJ: fsgnj.s s1, a0, a1
# CHECK-ASM: encoding: [0xd3,0x04,0xb5,0x20]
fsgnj.s s1, a0, a1
# CHECK-ASM-AND-OBJ: fsgnjn.s a1, a3, a4
# CHECK-ASM: encoding: [0xd3,0x95,0xe6,0x20]
fsgnjn.s a1, a3, a4
# CHECK-ASM-AND-OBJ: fsgnjx.s a4, a3, a2
# CHECK-ASM: encoding: [0x53,0xa7,0xc6,0x20]
fsgnjx.s a4, a3, a2
# CHECK-ASM-AND-OBJ: fmin.s a5, a6, a7
# CHECK-ASM: encoding: [0xd3,0x07,0x18,0x29]
fmin.s a5, a6, a7
# CHECK-ASM-AND-OBJ: fmax.s s2, s3, s4
# CHECK-ASM: encoding: [0x53,0x99,0x49,0x29]
fmax.s s2, s3, s4
# CHECK-ASM-AND-OBJ: fcvt.w.s a0, s5, dyn
# CHECK-ASM: encoding: [0x53,0xf5,0x0a,0xc0]
fcvt.w.s a0, s5, dyn
# CHECK-ASM-AND-OBJ: fcvt.wu.s a1, s6, dyn
# CHECK-ASM: encoding: [0xd3,0x75,0x1b,0xc0]
fcvt.wu.s a1, s6, dyn
# CHECK-ASM-AND-OBJ: feq.s a1, s8, s9
# CHECK-ASM: encoding: [0xd3,0x25,0x9c,0xa1]
feq.s a1, s8, s9
# CHECK-ASM-AND-OBJ: flt.s a2, s10, s11
# CHECK-ASM: encoding: [0x53,0x16,0xbd,0xa1]
flt.s a2, s10, s11
# CHECK-ASM-AND-OBJ: fle.s a3, t3, t4
# CHECK-ASM: encoding: [0xd3,0x06,0xde,0xa1]
fle.s a3, t3, t4
# CHECK-ASM-AND-OBJ: fclass.s a3, t5
# CHECK-ASM: encoding: [0xd3,0x16,0x0f,0xe0]
fclass.s a3, t5
# CHECK-ASM-AND-OBJ: fcvt.s.w t6, a4, dyn
# CHECK-ASM: encoding: [0xd3,0x7f,0x07,0xd0]
fcvt.s.w t6, a4, dyn
# CHECK-ASM-AND-OBJ: fcvt.s.wu s0, a5, dyn
# CHECK-ASM: encoding: [0x53,0xf4,0x17,0xd0]
fcvt.s.wu s0, a5, dyn

# Rounding modes

# CHECK-ASM-AND-OBJ: fmadd.s a0, a1, a2, a3, rne
# CHECK-ASM: encoding: [0x43,0x85,0xc5,0x68]
fmadd.s x10, x11, x12, x13, rne
# CHECK-ASM-AND-OBJ: fmsub.s a4, a5, a6, a7, rtz
# CHECK-ASM: encoding: [0x47,0x97,0x07,0x89]
fmsub.s x14, x15, x16, x17, rtz
# CHECK-ASM-AND-OBJ: fnmsub.s s2, s3, s4, s5, rdn
# CHECK-ASM: encoding: [0x4b,0xa9,0x49,0xa9]
fnmsub.s x18, x19, x20, x21, rdn
# CHECK-ASM-AND-OBJ: fnmadd.s s6, s7, s8, s9, rup
# CHECK-ASM: encoding: [0x4f,0xbb,0x8b,0xc9]
fnmadd.s x22, x23, x24, x25, rup
# CHECK-ASM-AND-OBJ: fmadd.s a0, a1, a2, a3, rmm
# CHECK-ASM: encoding: [0x43,0xc5,0xc5,0x68]
fmadd.s x10, x11, x12, x13, rmm
# CHECK-ASM-AND-OBJ: fmsub.s a4, a5, a6, a7
# CHECK-ASM: encoding: [0x47,0xf7,0x07,0x89]
fmsub.s x14, x15, x16, x17, dyn

# CHECK-ASM-AND-OBJ: fadd.s s10, s11, t3, rne
# CHECK-ASM: encoding: [0x53,0x8d,0xcd,0x01]
fadd.s x26, x27, x28, rne
# CHECK-ASM-AND-OBJ: fsub.s t4, t5, t6, rtz
# CHECK-ASM: encoding: [0xd3,0x1e,0xff,0x09]
fsub.s x29, x30, x31, rtz
# CHECK-ASM-AND-OBJ: fmul.s s0, s1, s2, rdn
# CHECK-ASM: encoding: [0x53,0xa4,0x24,0x11]
fmul.s s0, s1, s2, rdn
# CHECK-ASM-AND-OBJ: fdiv.s s3, s4, s5, rup
# CHECK-ASM: encoding: [0xd3,0x39,0x5a,0x19]
fdiv.s s3, s4, s5, rup

# CHECK-ASM-AND-OBJ: fsqrt.s t1, t2, rmm
# CHECK-ASM: encoding: [0x53,0xc3,0x03,0x58]
fsqrt.s t1, t2, rmm
# CHECK-ASM-AND-OBJ: fcvt.w.s a0, s5, rup
# CHECK-ASM: encoding: [0x53,0xb5,0x0a,0xc0]
fcvt.w.s a0, s5, rup
# CHECK-ASM-AND-OBJ: fcvt.wu.s a1, s6, rdn
# CHECK-ASM: encoding: [0xd3,0x25,0x1b,0xc0]
fcvt.wu.s a1, s6, rdn
# CHECK-ASM-AND-OBJ: fcvt.s.w t6, a4, rtz
# CHECK-ASM: encoding: [0xd3,0x1f,0x07,0xd0]
fcvt.s.w t6, a4, rtz
# CHECK-ASM-AND-OBJ: fcvt.s.wu s0, a5, rne
# CHECK-ASM: encoding: [0x53,0x84,0x17,0xd0]
fcvt.s.wu s0, a5, rne
