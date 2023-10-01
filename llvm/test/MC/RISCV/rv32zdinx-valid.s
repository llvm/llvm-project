# RUN: llvm-mc %s -triple=riscv32 -mattr=+zdinx -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zdinx %s \
# RUN:     | llvm-objdump --mattr=+zdinx -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zdinx -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zdinx %s \
# RUN:     | llvm-objdump --mattr=+zdinx -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: fmadd.d a0, a2, a4, a6, dyn
# CHECK-ASM: encoding: [0x43,0x75,0xe6,0x82]
fmadd.d x10, x12, x14, x16, dyn
# CHECK-ASM-AND-OBJ: fmsub.d a0, a2, a4, a6, dyn
# CHECK-ASM: encoding: [0x47,0x75,0xe6,0x82]
fmsub.d x10, x12, x14, x16, dyn
# CHECK-ASM-AND-OBJ: fnmsub.d a0, a2, a4, a6, dyn
# CHECK-ASM: encoding: [0x4b,0x75,0xe6,0x82]
fnmsub.d x10, x12, x14, x16, dyn
# CHECK-ASM-AND-OBJ: fnmadd.d a0, a2, a4, a6, dyn
# CHECK-ASM: encoding: [0x4f,0x75,0xe6,0x82]
fnmadd.d x10, x12, x14, x16, dyn

# CHECK-ASM-AND-OBJ: fadd.d s10, t3, t5, dyn
# CHECK-ASM: encoding: [0x53,0x7d,0xee,0x03]
fadd.d x26, x28, x30, dyn
# CHECK-ASM-AND-OBJ: fsub.d s10, t3, t5, dyn
# CHECK-ASM: encoding: [0x53,0x7d,0xee,0x0b]
fsub.d x26, x28, x30, dyn
# CHECK-ASM-AND-OBJ: fmul.d s10, t3, t5, dyn
# CHECK-ASM: encoding: [0x53,0x7d,0xee,0x13]
fmul.d x26, x28, x30, dyn
# CHECK-ASM-AND-OBJ: fdiv.d s10, t3, t5, dyn
# CHECK-ASM: encoding: [0x53,0x7d,0xee,0x1b]
fdiv.d x26, x28, x30, dyn
# CHECK-ASM-AND-OBJ: fsqrt.d s4, s6, dyn
# CHECK-ASM: encoding: [0x53,0x7a,0x0b,0x5a]
fsqrt.d x20, x22, dyn
# CHECK-ASM-AND-OBJ: fsgnj.d s10, t3, t5
# CHECK-ASM: encoding: [0x53,0x0d,0xee,0x23]
fsgnj.d x26, x28, x30
# CHECK-ASM-AND-OBJ: fsgnjn.d s10, t3, t5
# CHECK-ASM: encoding: [0x53,0x1d,0xee,0x23]
fsgnjn.d x26, x28, x30
# CHECK-ASM-AND-OBJ: fsgnjx.d s10, t3, t5
# CHECK-ASM: encoding: [0x53,0x2d,0xee,0x23]
fsgnjx.d x26, x28, x30
# CHECK-ASM-AND-OBJ: fmin.d s10, t3, t5
# CHECK-ASM: encoding: [0x53,0x0d,0xee,0x2b]
fmin.d x26, x28, x30
# CHECK-ASM-AND-OBJ: fmax.d s10, t3, t5
# CHECK-ASM: encoding: [0x53,0x1d,0xee,0x2b]
fmax.d x26, x28, x30

# CHECK-ASM-AND-OBJ: fcvt.s.d s10, t3, dyn
# CHECK-ASM: encoding: [0x53,0x7d,0x1e,0x40]
fcvt.s.d x26, x28, dyn
# CHECK-ASM-AND-OBJ: fcvt.d.s s10, t3
# CHECK-ASM: encoding: [0x53,0x0d,0x0e,0x42]
fcvt.d.s x26, x28
# CHECK-ASM-AND-OBJ: fcvt.d.s s10, t3, rup
# CHECK-ASM: encoding: [0x53,0x3d,0x0e,0x42]
fcvt.d.s x26, x28, rup
# CHECK-ASM-AND-OBJ: feq.d s10, t3, t5
# CHECK-ASM: encoding: [0x53,0x2d,0xee,0xa3]
feq.d x26, x28, x30
# CHECK-ASM-AND-OBJ: flt.d s10, t3, t5
# CHECK-ASM: encoding: [0x53,0x1d,0xee,0xa3]
flt.d x26, x28, x30
# CHECK-ASM-AND-OBJ: fle.d s10, t3, t5
# CHECK-ASM: encoding: [0x53,0x0d,0xee,0xa3]
fle.d x26, x28, x30
# CHECK-ASM-AND-OBJ: fclass.d s10, t3
# CHECK-ASM: encoding: [0x53,0x1d,0x0e,0xe2]
fclass.d x26, x28

# CHECK-ASM-AND-OBJ: fcvt.w.d s4, s6, dyn
# CHECK-ASM: encoding: [0x53,0x7a,0x0b,0xc2]
fcvt.w.d x20, x22, dyn
# CHECK-ASM-AND-OBJ: fcvt.d.w s10, t3
# CHECK-ASM: encoding: [0x53,0x0d,0x0e,0xd2]
fcvt.d.w x26, x28
# CHECK-ASM-AND-OBJ: fcvt.d.w s10, t3, rup
# CHECK-ASM: encoding: [0x53,0x3d,0x0e,0xd2]
fcvt.d.w x26, x28, rup
# CHECK-ASM-AND-OBJ: fcvt.d.wu s10, t3
# CHECK-ASM: encoding: [0x53,0x0d,0x1e,0xd2]
fcvt.d.wu x26, x28
# CHECK-ASM-AND-OBJ: fcvt.d.wu s10, t3, rup
# CHECK-ASM: encoding: [0x53,0x3d,0x1e,0xd2]
fcvt.d.wu x26, x28, rup

# Rounding modes

# CHECK-ASM-AND-OBJ: fmadd.d a0, a2, a4, a6, rne
# CHECK-ASM: encoding: [0x43,0x05,0xe6,0x82]
fmadd.d x10, x12, x14, x16, rne
# CHECK-ASM-AND-OBJ: fmsub.d a0, a2, a4, a6, rtz
# CHECK-ASM: encoding: [0x47,0x15,0xe6,0x82]
fmsub.d x10, x12, x14, x16, rtz
# CHECK-ASM-AND-OBJ: fnmsub.d a0, a2, a4, a6, rdn
# CHECK-ASM: encoding: [0x4b,0x25,0xe6,0x82]
fnmsub.d x10, x12, x14, x16, rdn
# CHECK-ASM-AND-OBJ: fnmadd.d a0, a2, a4, a6, rup
# CHECK-ASM: encoding: [0x4f,0x35,0xe6,0x82]
fnmadd.d x10, x12, x14, x16, rup

# CHECK-ASM-AND-OBJ: fadd.d s10, t3, t5, rmm
# CHECK-ASM: encoding: [0x53,0x4d,0xee,0x03]
fadd.d x26, x28, x30, rmm
# CHECK-ASM-AND-OBJ: fsub.d s10, t3, t5, dyn
# CHECK-ASM: encoding: [0x53,0x7d,0xee,0x0b]
fsub.d x26, x28, x30, dyn
# CHECK-ASM-AND-OBJ: fmul.d s10, t3, t5, rne
# CHECK-ASM: encoding: [0x53,0x0d,0xee,0x13]
fmul.d x26, x28, x30, rne
# CHECK-ASM-AND-OBJ: fdiv.d s10, t3, t5, rtz
# CHECK-ASM: encoding: [0x53,0x1d,0xee,0x1b]
fdiv.d x26, x28, x30, rtz

# CHECK-ASM-AND-OBJ: fsqrt.d s4, s6, rdn
# CHECK-ASM: encoding: [0x53,0x2a,0x0b,0x5a]
fsqrt.d x20, x22, rdn
# CHECK-ASM-AND-OBJ: fcvt.s.d s4, s6, rup
# CHECK-ASM: encoding: [0x53,0x3a,0x1b,0x40]
fcvt.s.d x20, x22, rup
# CHECK-ASM-AND-OBJ: fcvt.w.d s4, s6, rmm
# CHECK-ASM: encoding: [0x53,0x4a,0x0b,0xc2]
fcvt.w.d x20, x22, rmm
# CHECK-ASM-AND-OBJ: fcvt.wu.d s4, s6, dyn
# CHECK-ASM: encoding: [0x53,0x7a,0x1b,0xc2]
fcvt.wu.d x20, x22, dyn
