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

## Support for the 'D' extension implies support for 'F'
# ASM-AND-OBJ: fld.s $ft15, $t3, 250
# ASM: encoding: [0xf7,0xe9,0x03,0x2b]
fld.s $ft15, $t3, 250

# ASM-AND-OBJ: fld.d $ft14, $t5, 114
# ASM: encoding: [0x36,0xca,0x81,0x2b]
fld.d $ft14, $t5, 114

# ASM-AND-OBJ: fst.d $fs4, $a3, 198
# ASM: encoding: [0xfc,0x18,0xc3,0x2b]
fst.d $fs4, $a3, 198

# ASM-AND-OBJ: fldx.d $fs3, $t1, $s8
# ASM: encoding: [0xbb,0x7d,0x34,0x38]
fldx.d $fs3, $t1, $s8

# ASM-AND-OBJ: fstx.d $fa6, $t3, $t5
# ASM: encoding: [0xe6,0x45,0x3c,0x38]
fstx.d $fa6, $t3, $t5
