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

# ASM-AND-OBJ: fld.s $ft15, $t3, 250
# ASM: encoding: [0xf7,0xe9,0x03,0x2b]
fld.s $ft15, $t3, 250

# ASM-AND-OBJ: fst.s $fs6, $t7, 230
# ASM: encoding: [0x7e,0x9a,0x43,0x2b]
fst.s $fs6, $t7, 230

# ASM-AND-OBJ: fldx.s $fa1, $t3, $t7
# ASM: encoding: [0xe1,0x4d,0x30,0x38]
fldx.s $fa1, $t3, $t7

# ASM-AND-OBJ: fstx.s $fs2, $sp, $fp
# ASM: encoding: [0x7a,0x58,0x38,0x38]
fstx.s $fs2, $sp, $fp
