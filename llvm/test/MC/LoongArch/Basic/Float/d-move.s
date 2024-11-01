# RUN: llvm-mc %s --triple=loongarch32 --mattr=+d --show-encoding \
# RUN:     | FileCheck --check-prefixes=ASM-AND-OBJ,ASM %s
# RUN: llvm-mc %s --triple=loongarch64 --mattr=+d --show-encoding --defsym=LA64=1 \
# RUN:     | FileCheck --check-prefixes=ASM-AND-OBJ,ASM,ASM-AND-OBJ64,ASM64 %s
# RUN: llvm-mc %s --triple=loongarch32 --mattr=+d --filetype=obj \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck --check-prefix=ASM-AND-OBJ %s
# RUN: llvm-mc %s --triple=loongarch64 --mattr=+d --filetype=obj --defsym=LA64=1 \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck --check-prefixes=ASM-AND-OBJ,ASM-AND-OBJ64 %s

## Support for the 'D' extension implies support for 'F'
# ASM-AND-OBJ: fmov.s $ft5, $ft15
# ASM: encoding: [0xed,0x96,0x14,0x01]
fmov.s $ft5, $ft15

# ASM-AND-OBJ: fmov.d $fs6, $ft1
# ASM: encoding: [0x3e,0x99,0x14,0x01]
fmov.d $fs6, $ft1

# ASM-AND-OBJ: fsel $ft10, $ft12, $ft13, $fcc4
# ASM: encoding: [0x92,0x56,0x02,0x0d]
fsel $ft10, $ft12, $ft13, $fcc4

# ASM-AND-OBJ64: movgr2frh.w $ft15, $s3
# ASM64: encoding: [0x57,0xaf,0x14,0x01]
movgr2frh.w $ft15, $s3

.ifdef LA64

# ASM-AND-OBJ64: movgr2fr.d $fs6, $a7
# ASM64: encoding: [0x7e,0xa9,0x14,0x01]
movgr2fr.d $fs6, $a7

# ASM-AND-OBJ64: movfr2gr.d $s3, $ft9
# ASM64: encoding: [0x3a,0xba,0x14,0x01]
movfr2gr.d $s3, $ft9

.endif
