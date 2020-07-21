# RUN: llvm-mc %s -arch=mips -mcpu=mips32r2 -filetype=obj -o - | \
# RUN:   llvm-readobj -S --section-data - | \
# RUN:     FileCheck %s

# CHECK:   Section {
# CHECK:    Index:
# CHECK:    Name: .reginfo
# CHECK:    Type: SHT_MIPS_REGINFO (0x70000006)
# CHECK:    Flags [ (0x2)
# CHECK:      SHF_ALLOC (0x2)
# CHECK:    ]
# CHECK:    Size: 24
# CHECK:    SectionData (
# CHECK:      0000: 01010101 00000000 C0007535 00000000
# CHECK:      0010: 00000000 00000000
# CHECK:    )
# CHECK:  }

.text
        add $0,$0,$0
        add $8,$0,$0
        add $16,$0,$0
        add $24,$0,$0

# abs.s - Reads and writes from/to $f0.
        abs.s $f0,$f0
# round.w.d - Reads $f4 and $f5 and writes to $f2.
        round.w.d $f2,$f4
# ceil.w.s - Reads $f8 and writes to $f10.
        ceil.w.s $f10, $f8
# cvt.s.d - Reads from $f12 and $f13 and writes to $f14
        cvt.s.d $f14, $f12
# abs.d - Reads from $f30 and $f31 and writes to $f30 and $f31.
        abs.d $f30,$f30
