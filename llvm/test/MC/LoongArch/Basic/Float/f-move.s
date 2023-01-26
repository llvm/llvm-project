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

# ASM-AND-OBJ: fmov.s $ft5, $ft15
# ASM: encoding: [0xed,0x96,0x14,0x01]
fmov.s $ft5, $ft15

# ASM-AND-OBJ: fsel $ft10, $ft12, $ft13, $fcc4
# ASM: encoding: [0x92,0x56,0x02,0x0d]
fsel $ft10, $ft12, $ft13, $fcc4

# ASM-AND-OBJ: movgr2fr.w $fa6, $tp
# ASM: encoding: [0x46,0xa4,0x14,0x01]
movgr2fr.w $fa6, $tp

# ASM-AND-OBJ: movfr2gr.s $a6, $ft14
# ASM: encoding: [0xca,0xb6,0x14,0x01]
movfr2gr.s $a6, $ft14

# ASM-AND-OBJ: movgr2fcsr $fcsr0, $a0
# ASM: encoding: [0x80,0xc0,0x14,0x01]
movgr2fcsr $fcsr0, $a0

# ASM-AND-OBJ: movfcsr2gr $a0, $fcsr0
# ASM: encoding: [0x04,0xc8,0x14,0x01]
movfcsr2gr $a0, $fcsr0

# ASM-AND-OBJ: movgr2fcsr $fcsr1, $a0
# ASM: encoding: [0x81,0xc0,0x14,0x01]
movgr2fcsr $fcsr1, $a0

# ASM-AND-OBJ: movfcsr2gr $a0, $fcsr1
# ASM: encoding: [0x24,0xc8,0x14,0x01]
movfcsr2gr $a0, $fcsr1

# ASM-AND-OBJ: movgr2fcsr $fcsr2, $a0
# ASM: encoding: [0x82,0xc0,0x14,0x01]
movgr2fcsr $fcsr2, $a0

# ASM-AND-OBJ: movfcsr2gr $a0, $fcsr2
# ASM: encoding: [0x44,0xc8,0x14,0x01]
movfcsr2gr $a0, $fcsr2

# ASM-AND-OBJ: movgr2fcsr $fcsr3, $a0
# ASM: encoding: [0x83,0xc0,0x14,0x01]
movgr2fcsr $fcsr3, $a0

# ASM-AND-OBJ: movfcsr2gr $a0, $fcsr3
# ASM: encoding: [0x64,0xc8,0x14,0x01]
movfcsr2gr $a0, $fcsr3

# ASM-AND-OBJ: movfr2cf $fcc4, $ft3
# ASM: encoding: [0x64,0xd1,0x14,0x01]
movfr2cf $fcc4, $ft3

# ASM-AND-OBJ: movcf2fr $ft8, $fcc0
# ASM: encoding: [0x10,0xd4,0x14,0x01]
movcf2fr $ft8, $fcc0

# ASM-AND-OBJ: movgr2cf $fcc5, $ra
# ASM: encoding: [0x25,0xd8,0x14,0x01]
movgr2cf $fcc5, $ra

# ASM-AND-OBJ: movcf2gr $r21, $fcc7
# ASM: encoding: [0xf5,0xdc,0x14,0x01]
movcf2gr $r21, $fcc7
