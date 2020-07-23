# RUN: llvm-mc %s -arch=mips -mcpu=mips32r2 -mattr=+msa,+fp64 -filetype=obj -o - | \
# RUN:   llvm-readobj -S --section-data - | \
# RUN:     FileCheck %s -check-prefix=ELF32

# RUN: llvm-mc %s -arch=mips64 -mcpu=mips64r2 -mattr=+msa,+fp64 -target-abi n32 -filetype=obj -o - | \
# RUN:   llvm-readobj -S --section-data - | \
# RUN:     FileCheck %s -check-prefix=ELF32

# RUN: llvm-mc %s -arch=mips64 -mcpu=mips64r2 -mattr=+msa,+fp64 -target-abi n64 -filetype=obj -o - | \
# RUN:   llvm-readobj -S --section-data - | \
# RUN:     FileCheck %s -check-prefix=ELF64

# ELF32:   Section {
# ELF32:    Name: .reginfo
# ELF32:    Type: SHT_MIPS_REGINFO (0x70000006)
# ELF32:    Flags [ (0x2)
# ELF32:      SHF_ALLOC (0x2)
# ELF32:    ]
# ELF32:    Size: 24
# ELF32:    SectionData (
# ELF32:      0000: 01010101 00000000 4C005515 00000000
# ELF32:      0010: 00000000 00000000
# ELF32:    )
# ELF32:  }

# ELF64:   Section {
# ELF64:    Name: .MIPS.options
# ELF64:    Type: SHT_MIPS_OPTIONS (0x7000000D)
# ELF64:    Flags [ (0x8000002)
# ELF64:      SHF_ALLOC (0x2)
# ELF64:      SHF_MIPS_NOSTRIP (0x8000000)
# ELF64:    ]
# ELF64:    Size: 40
# ELF64:    SectionData (
# ELF64:      0000: 01280000 00000000 01010101 00000000
# ELF64:      0010: 00000000 4C005515 00000000 00000000
# ELF64:      0020: 00000000 00000000
# ELF64:    )
# ELF64:  }

.text
        add $0,$0,$0
        add $8,$0,$0
        add $16,$0,$0
        add $24,$0,$0

# abs.s - Reads and writes from/to $f0.
        abs.s $f0,$f0
# round.w.d - Reads $f4 and writes to $f2.
        round.w.d $f2,$f4
# ceil.w.s - Reads $f8 and writes to $f10.
        ceil.w.s $f10, $f8
# cvt.s.d - Reads from $f12 and writes to $f14.
        cvt.s.d $f14, $f12
# abs.d - Reads from $f30 and writes to $f30.
        abs.d $f30,$f30

# Read and write from/to $f26 and $f27
        add_a.b $w26,$w26,$w26
        add_a.b $w27,$w27,$w27
