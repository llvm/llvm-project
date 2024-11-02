# RUN: llvm-mc %s --triple=loongarch32 --mattr=+d --show-encoding \
# RUN:     | FileCheck --check-prefixes=ASM-AND-OBJ,ASM %s
# RUN: llvm-mc %s --triple=loongarch64 --mattr=+d --show-encoding \
# RUN:     | FileCheck --check-prefixes=ASM-AND-OBJ,ASM %s
# RUN: llvm-mc %s --triple=loongarch32 --mattr=+d --filetype=obj \
# RUN:     | llvm-objdump -d --mattr=+d - \
# RUN:     | FileCheck --check-prefix=ASM-AND-OBJ %s
# RUN: llvm-mc %s --triple=loongarch64 --mattr=+d --filetype=obj \
# RUN:     | llvm-objdump -d --mattr=+d - \
# RUN:     | FileCheck --check-prefix=ASM-AND-OBJ %s

## Support for the 'D' extension implies support for 'F'
# ASM-AND-OBJ: fadd.s $fs5, $ft7, $fs1
# ASM: encoding: [0xfd,0xe5,0x00,0x01]
fadd.s $fs5, $ft7, $fs1

# ASM-AND-OBJ: fadd.d $fs1, $fa7, $ft5
# ASM: encoding: [0xf9,0x34,0x01,0x01]
fadd.d $fs1, $fa7, $ft5

# ASM-AND-OBJ: fsub.d $fs5, $fa1, $ft10
# ASM: encoding: [0x3d,0x48,0x03,0x01]
fsub.d $fs5, $fa1, $ft10

# ASM-AND-OBJ: fmul.d $fa4, $fs6, $fa7
# ASM: encoding: [0xc4,0x1f,0x05,0x01]
fmul.d $fa4, $fs6, $fa7

# ASM-AND-OBJ: fdiv.d $fa3, $fs1, $fs4
# ASM: encoding: [0x23,0x73,0x07,0x01]
fdiv.d $fa3, $fs1, $fs4

# ASM-AND-OBJ: fmadd.d $ft13, $fs0, $fs4, $fs0
# ASM: encoding: [0x15,0x73,0x2c,0x08]
fmadd.d $ft13, $fs0, $fs4, $fs0

# ASM-AND-OBJ: fmsub.d $fa6, $ft10, $ft12, $fs3
# ASM: encoding: [0x46,0xd2,0x6d,0x08]
fmsub.d $fa6, $ft10, $ft12, $fs3

# ASM-AND-OBJ: fnmadd.d $fs1, $ft5, $ft11, $fs6
# ASM: encoding: [0xb9,0x4d,0xaf,0x08]
fnmadd.d $fs1, $ft5, $ft11, $fs6

# ASM-AND-OBJ: fnmsub.d $fs6, $fs2, $fa7, $fs0
# ASM: encoding: [0x5e,0x1f,0xec,0x08]
fnmsub.d $fs6, $fs2, $fa7, $fs0

# ASM-AND-OBJ: fmax.d $ft3, $fs2, $ft5
# ASM: encoding: [0x4b,0x37,0x09,0x01]
fmax.d $ft3, $fs2, $ft5

# ASM-AND-OBJ: fmin.d $fa1, $ft5, $fs3
# ASM: encoding: [0xa1,0x6d,0x0b,0x01]
fmin.d $fa1, $ft5, $fs3

# ASM-AND-OBJ: fmaxa.d $fs0, $ft5, $fa4
# ASM: encoding: [0xb8,0x11,0x0d,0x01]
fmaxa.d $fs0, $ft5, $fa4

# ASM-AND-OBJ: fmina.d $ft10, $ft2, $fa0
# ASM: encoding: [0x52,0x01,0x0f,0x01]
fmina.d $ft10, $ft2, $fa0

# ASM-AND-OBJ: fabs.d $ft15, $fa3
# ASM: encoding: [0x77,0x08,0x14,0x01]
fabs.d $ft15, $fa3

# ASM-AND-OBJ: fneg.d $ft3, $fs2
# ASM: encoding: [0x4b,0x1b,0x14,0x01]
fneg.d $ft3, $fs2

# ASM-AND-OBJ: fsqrt.d $fa2, $ft3
# ASM: encoding: [0x62,0x49,0x14,0x01]
fsqrt.d $fa2, $ft3

# ASM-AND-OBJ: frecip.d $fs3, $fs3
# ASM: encoding: [0x7b,0x5b,0x14,0x01]
frecip.d $fs3, $fs3

# ASM-AND-OBJ: frsqrt.d $ft14, $fa3
# ASM: encoding: [0x76,0x68,0x14,0x01]
frsqrt.d $ft14, $fa3

# ASM-AND-OBJ: fscaleb.d $ft4, $ft6, $fs2
# ASM: encoding: [0xcc,0x69,0x11,0x01]
fscaleb.d $ft4, $ft6, $fs2

# ASM-AND-OBJ: flogb.d $ft13, $fs5
# ASM: encoding: [0xb5,0x2b,0x14,0x01]
flogb.d $ft13, $fs5

# ASM-AND-OBJ: fcopysign.d $ft8, $fs2, $fa6
# ASM: encoding: [0x50,0x1b,0x13,0x01]
fcopysign.d $ft8, $fs2, $fa6

# ASM-AND-OBJ: fclass.d $ft11, $fa2
# ASM: encoding: [0x53,0x38,0x14,0x01]
fclass.d $ft11, $fa2
