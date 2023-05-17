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

# ASM-AND-OBJ: ffint.s.w $fs6, $fa5
# ASM: encoding: [0xbe,0x10,0x1d,0x01]
ffint.s.w $fs6, $fa5

# ASM-AND-OBJ: ftint.w.s $ft13, $ft5
# ASM: encoding: [0xb5,0x05,0x1b,0x01]
ftint.w.s $ft13, $ft5

# ASM-AND-OBJ: ftintrm.w.s $ft8, $ft8
# ASM: encoding: [0x10,0x06,0x1a,0x01]
ftintrm.w.s $ft8, $ft8

# ASM-AND-OBJ: ftintrp.w.s $ft6, $fs7
# ASM: encoding: [0xee,0x47,0x1a,0x01]
ftintrp.w.s $ft6, $fs7

# ASM-AND-OBJ: ftintrz.w.s $fa4, $fs5
# ASM: encoding: [0xa4,0x87,0x1a,0x01]
ftintrz.w.s $fa4, $fs5

# ASM-AND-OBJ: ftintrne.w.s $fa4, $ft9
# ASM: encoding: [0x24,0xc6,0x1a,0x01]
ftintrne.w.s $fa4, $ft9

# ASM-AND-OBJ: frint.s $fa5, $ft9
# ASM: encoding: [0x25,0x46,0x1e,0x01]
frint.s $fa5, $ft9
