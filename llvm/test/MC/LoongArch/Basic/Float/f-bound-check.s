# RUN: llvm-mc %s --triple=loongarch32 --mattr=+f --show-encoding \
# RUN:     | FileCheck --check-prefixes=ASM-AND-OBJ,ASM %s
# RUN: llvm-mc %s --triple=loongarch64 --mattr=+f --show-encoding \
# RUN:     | FileCheck --check-prefixes=ASM-AND-OBJ,ASM %s
# RUN: llvm-mc %s --triple=loongarch32 --mattr=+f --filetype=obj \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck --check-prefix=ASM-AND-OBJ %s
# RUN: llvm-mc %s --triple=loongarch64 --mattr=+f --filetype=obj \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck --check-prefix=ASM-AND-OBJ %s

# ASM-AND-OBJ: fldgt.s $fa3, $s4, $t1
# ASM: encoding: [0x63,0x37,0x74,0x38]
fldgt.s $fa3, $s4, $t1

# ASM-AND-OBJ: fldle.s $fs0, $s6, $t5
# ASM: encoding: [0xb8,0x47,0x75,0x38]
fldle.s $fs0, $s6, $t5

# ASM-AND-OBJ: fstgt.s $fs7, $t1, $s7
# ASM: encoding: [0xbf,0x79,0x76,0x38]
fstgt.s $fs7, $t1, $s7

# ASM-AND-OBJ: fstle.s $ft5, $t1, $a3
# ASM: encoding: [0xad,0x1d,0x77,0x38]
fstle.s $ft5, $t1, $a3
