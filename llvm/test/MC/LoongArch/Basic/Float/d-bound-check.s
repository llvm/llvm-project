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
# ASM-AND-OBJ: fldgt.s $fa3, $s4, $t1
# ASM: encoding: [0x63,0x37,0x74,0x38]
fldgt.s $fa3, $s4, $t1

# ASM-AND-OBJ: fldgt.d $fs2, $a1, $s8
# ASM: encoding: [0xba,0xfc,0x74,0x38]
fldgt.d $fs2, $a1, $s8

# ASM-AND-OBJ: fldle.d $fa3, $t3, $fp
# ASM: encoding: [0xe3,0xd9,0x75,0x38]
fldle.d $fa3, $t3, $fp

# ASM-AND-OBJ: fstgt.d $ft5, $a7, $s3
# ASM: encoding: [0x6d,0xe9,0x76,0x38]
fstgt.d $ft5, $a7, $s3

# ASM-AND-OBJ: fstle.d $ft10, $a5, $t1
# ASM: encoding: [0x32,0xb5,0x77,0x38]
fstle.d $ft10, $a5, $t1
