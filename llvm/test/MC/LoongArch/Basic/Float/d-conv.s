# RUN: llvm-mc %s --triple=loongarch32 --mattr=+d --show-encoding \
# RUN:     | FileCheck --check-prefixes=ASM-AND-OBJ,ASM %s
# RUN: llvm-mc %s --triple=loongarch64 --mattr=+d --show-encoding \
# RUN:     | FileCheck --check-prefixes=ASM-AND-OBJ,ASM %s
# RUN: llvm-mc %s --triple=loongarch32 --mattr=+d --filetype=obj \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck --check-prefix=ASM-AND-OBJ %s
# RUN: llvm-mc %s --triple=loongarch64 --mattr=+d --filetype=obj \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck --check-prefix=ASM-AND-OBJ %s

## Support for the 'D' extension implies support for 'F'
# ASM-AND-OBJ: frint.s $fa5, $ft9
# ASM: encoding: [0x25,0x46,0x1e,0x01]
frint.s $fa5, $ft9

# ASM-AND-OBJ: fcvt.s.d $ft4, $ft11
# ASM: encoding: [0x6c,0x1a,0x19,0x01]
fcvt.s.d $ft4, $ft11

# ASM-AND-OBJ: fcvt.d.s $ft2, $fa6
# ASM: encoding: [0xca,0x24,0x19,0x01]
fcvt.d.s $ft2, $fa6

# ASM-AND-OBJ: ffint.s.l $fa6, $fa5
# ASM: encoding: [0xa6,0x18,0x1d,0x01]
ffint.s.l $fa6, $fa5

# ASM-AND-OBJ: ffint.d.w $fs0, $ft10
# ASM: encoding: [0x58,0x22,0x1d,0x01]
ffint.d.w $fs0, $ft10

# ASM-AND-OBJ: ffint.d.l $ft15, $fs2
# ASM: encoding: [0x57,0x2b,0x1d,0x01]
ffint.d.l $ft15, $fs2

# ASM-AND-OBJ: ftint.w.d $fa3, $ft6
# ASM: encoding: [0xc3,0x09,0x1b,0x01]
ftint.w.d $fa3, $ft6

# ASM-AND-OBJ: ftint.l.s $fs7, $fs0
# ASM: encoding: [0x1f,0x27,0x1b,0x01]
ftint.l.s $fs7, $fs0

# ASM-AND-OBJ: ftint.l.d $ft8, $fs0
# ASM: encoding: [0x10,0x2b,0x1b,0x01]
ftint.l.d $ft8, $fs0

# ASM-AND-OBJ: ftintrm.w.d $fa7, $ft0
# ASM: encoding: [0x07,0x09,0x1a,0x01]
ftintrm.w.d $fa7, $ft0

# ASM-AND-OBJ: ftintrm.l.s $fs0, $ft2
# ASM: encoding: [0x58,0x25,0x1a,0x01]
ftintrm.l.s $fs0, $ft2

# ASM-AND-OBJ: ftintrm.l.d $ft1, $ft1
# ASM: encoding: [0x29,0x29,0x1a,0x01]
ftintrm.l.d $ft1, $ft1

# ASM-AND-OBJ: ftintrp.w.d $ft4, $fa3
# ASM: encoding: [0x6c,0x48,0x1a,0x01]
ftintrp.w.d $ft4, $fa3

# ASM-AND-OBJ: ftintrp.l.s $fa0, $ft8
# ASM: encoding: [0x00,0x66,0x1a,0x01]
ftintrp.l.s $fa0, $ft8

# ASM-AND-OBJ: ftintrp.l.d $fa4, $fs5
# ASM: encoding: [0xa4,0x6b,0x1a,0x01]
ftintrp.l.d $fa4, $fs5

# ASM-AND-OBJ: ftintrz.w.d $fs1, $fs0
# ASM: encoding: [0x19,0x8b,0x1a,0x01]
ftintrz.w.d $fs1, $fs0

# ASM-AND-OBJ: ftintrz.l.s $ft15, $fa5
# ASM: encoding: [0xb7,0xa4,0x1a,0x01]
ftintrz.l.s $ft15, $fa5

# ASM-AND-OBJ: ftintrz.l.d $fa3, $ft2
# ASM: encoding: [0x43,0xa9,0x1a,0x01]
ftintrz.l.d $fa3, $ft2

# ASM-AND-OBJ: ftintrne.w.d $fs7, $ft4
# ASM: encoding: [0x9f,0xc9,0x1a,0x01]
ftintrne.w.d $fs7, $ft4

# ASM-AND-OBJ: ftintrne.l.s $ft14, $fs3
# ASM: encoding: [0x76,0xe7,0x1a,0x01]
ftintrne.l.s $ft14, $fs3

# ASM-AND-OBJ: ftintrne.l.d $fs4, $fa6
# ASM: encoding: [0xdc,0xe8,0x1a,0x01]
ftintrne.l.d $fs4, $fa6

# ASM-AND-OBJ: frint.d $fs5, $fa2
# ASM: encoding: [0x5d,0x48,0x1e,0x01]
frint.d $fs5, $fa2
