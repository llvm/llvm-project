# RUN: llvm-mc %s --triple=loongarch32 --show-encoding \
# RUN:     | FileCheck --check-prefixes=ASM-AND-OBJ,ASM %s
# RUN: llvm-mc %s --triple=loongarch64 --show-encoding \
# RUN:     | FileCheck --check-prefixes=ASM-AND-OBJ,ASM %s
# RUN: llvm-mc %s --triple=loongarch32 --filetype=obj \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck --check-prefix=ASM-AND-OBJ %s
# RUN: llvm-mc %s --triple=loongarch64 --filetype=obj \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck --check-prefix=ASM-AND-OBJ %s

# ASM-AND-OBJ: fadd.s $fs5, $ft7, $fs1
# ASM: encoding: [0xfd,0xe5,0x00,0x01]
fadd.s $fs5, $ft7, $fs1

# ASM-AND-OBJ: fsub.s $ft6, $fa6, $fs7
# ASM: encoding: [0xce,0xfc,0x02,0x01]
fsub.s $ft6, $fa6, $fs7

# ASM-AND-OBJ: fmul.s $fa0, $fa7, $ft9
# ASM: encoding: [0xe0,0xc4,0x04,0x01]
fmul.s $fa0, $fa7, $ft9

# ASM-AND-OBJ: fdiv.s $ft12, $fs0, $ft11
# ASM: encoding: [0x14,0xcf,0x06,0x01]
fdiv.s $ft12, $fs0, $ft11

# ASM-AND-OBJ: fmadd.s $fa3, $ft8, $fa3, $ft7
# ASM: encoding: [0x03,0x8e,0x17,0x08]
fmadd.s $fa3, $ft8, $fa3, $ft7

# ASM-AND-OBJ: fmsub.s $ft15, $ft3, $ft13, $fa4
# ASM: encoding: [0x77,0x55,0x52,0x08]
fmsub.s $ft15, $ft3, $ft13, $fa4

# ASM-AND-OBJ: fnmadd.s $fs5, $fa1, $fs0, $ft12
# ASM: encoding: [0x3d,0x60,0x9a,0x08]
fnmadd.s $fs5, $fa1, $fs0, $ft12

# ASM-AND-OBJ: fnmsub.s $ft0, $fa4, $fs0, $fs1
# ASM: encoding: [0x88,0xe0,0xdc,0x08]
fnmsub.s $ft0, $fa4, $fs0, $fs1

# ASM-AND-OBJ: fmax.s $ft14, $fa6, $fs3
# ASM: encoding: [0xd6,0xec,0x08,0x01]
fmax.s $ft14, $fa6, $fs3

# ASM-AND-OBJ: fmin.s $ft6, $ft2, $ft11
# ASM: encoding: [0x4e,0xcd,0x0a,0x01]
fmin.s $ft6, $ft2, $ft11

# ASM-AND-OBJ: fmaxa.s $ft1, $fs3, $fs7
# ASM: encoding: [0x69,0xff,0x0c,0x01]
fmaxa.s $ft1, $fs3, $fs7

# ASM-AND-OBJ: fmina.s $ft7, $ft10, $fa1
# ASM: encoding: [0x4f,0x86,0x0e,0x01]
fmina.s $ft7, $ft10, $fa1

# ASM-AND-OBJ: fabs.s $fs4, $ft4
# ASM: encoding: [0x9c,0x05,0x14,0x01]
fabs.s $fs4, $ft4

# ASM-AND-OBJ: fneg.s $ft13, $fs0
# ASM: encoding: [0x15,0x17,0x14,0x01]
fneg.s $ft13, $fs0

# ASM-AND-OBJ: fsqrt.s $fs3, $ft10
# ASM: encoding: [0x5b,0x46,0x14,0x01]
fsqrt.s $fs3, $ft10

# ASM-AND-OBJ: frecip.s $ft9, $fs3
# ASM: encoding: [0x71,0x57,0x14,0x01]
frecip.s $ft9, $fs3

# ASM-AND-OBJ: frsqrt.s $fs1, $ft4
# ASM: encoding: [0x99,0x65,0x14,0x01]
frsqrt.s $fs1, $ft4

# ASM-AND-OBJ: fscaleb.s $ft13, $ft15, $fa6
# ASM: encoding: [0xf5,0x9a,0x10,0x01]
fscaleb.s $ft13, $ft15, $fa6

# ASM-AND-OBJ: flogb.s $fs7, $ft15
# ASM: encoding: [0xff,0x26,0x14,0x01]
flogb.s $fs7, $ft15

# ASM-AND-OBJ: fcopysign.s $ft5, $fs0, $ft15
# ASM: encoding: [0x0d,0xdf,0x12,0x01]
fcopysign.s $ft5, $fs0, $ft15

# ASM-AND-OBJ: fclass.s $ft12, $ft1
# ASM: encoding: [0x34,0x35,0x14,0x01]
fclass.s $ft12, $ft1
