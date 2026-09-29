// Verify the .fix data section conveys the right offsets and the right relocations
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-win32 %s -o - | llvm-readobj -r --expand-relocs -S --section-data - | FileCheck %s --check-prefix=READOBJ

        .text
	.section	.rdata,"dr"
	.globl	g1                      # @g1
	.align	4
g1:
	.long	1                       # 0x1

	.globl	g2                      # @g2
	.align	4
g2:
	.long	2                       # 0x2

	.globl	g3                      # @g3
	.align	4
g3:
	.long	3                       # 0x3

	.globl	g4                      # @g4
	.align	4
g4:
	.long	4                       # 0x4

	.section	.fix,"dw"
	.globl	t1                      # @t1
	.align	8
t1:

	.globl	t2                      # @t2
	.align	8
t2:

	.globl	t3                      # @t3
	.align	8
t3:

	.globl	t4                      # @t4
	.align	4
t4:
	.long	g3-t4

	.globl	t5                      # @t5
	.align	4
t5:
	.long	g3@IMGREL

	.globl	t6                      # @t6
	.align	16
t6:
	.ascii	"\001\002"
	.zero	6
	.quad	256                     # 0x100
	.long	g3-(t6+16)
	.zero	4

.long foobar - .

// As an extension, we allow 64-bit label differences. They lower to
// IMAGE_REL_AMD64_REL32 because IMAGE_REL_AMD64_REL64 does not exist.
.quad foobar - .

// READOBJ:       Section {
// READOBJ:         Number:
// READOBJ:         Name: .fix (2E 66 69 78 00 00 00 00)
// READOBJ-NEXT:    VirtualSize: 0x0
// READOBJ-NEXT:    VirtualAddress: 0x0
// READOBJ-NEXT:    RawDataSize:
// READOBJ-NEXT:    PointerToRawData: 0xEC
// READOBJ-NEXT:    PointerToRelocations:
// READOBJ-NEXT:    PointerToLineNumbers: 0x0
// READOBJ-NEXT:    RelocationCount:
// READOBJ-NEXT:    LineNumberCount: 0
// READOBJ-NEXT:    Characteristics [ (0xC0500040)
// READOBJ-NEXT:      IMAGE_SCN_ALIGN_16BYTES (0x500000)
// READOBJ-NEXT:      IMAGE_SCN_CNT_INITIALIZED_DATA (0x40)
// READOBJ-NEXT:      IMAGE_SCN_MEM_READ (0x40000000)
// READOBJ-NEXT:      IMAGE_SCN_MEM_WRITE (0x80000000)
// READOBJ-NEXT:    ]
// READOBJ-NEXT:    SectionData (
// READOBJ-NEXT:      0000: 04000000 00000000 00000000 00000000  |
// READOBJ-NEXT:      0010: 01020000 00000000 00010000 00000000  |
// READOBJ-NEXT:      0020: 04000000 00000000 04000000 04000000  |
// READOBJ-NEXT:      0030: 00000000 |
// READOBJ-NEXT:    )
// READOBJ-NEXT:  }
// READOBJ-NEXT:  ]
// READOBJ-NEXT:  Relocations [
// READOBJ-NEXT:  Section (5) .fix {
// READOBJ-NEXT:    Relocation {
// READOBJ-NEXT:      Offset: 0x0
// READOBJ-NEXT:      Type: IMAGE_REL_AMD64_REL32 (4)
// READOBJ-NEXT:      Symbol: g3
// READOBJ-NEXT:      SymbolIndex: 12
// READOBJ-NEXT:    }
// READOBJ-NEXT:    Relocation {
// READOBJ-NEXT:      Offset: 0x4
// READOBJ-NEXT:      Type: IMAGE_REL_AMD64_ADDR32NB (3)
// READOBJ-NEXT:      Symbol: g3
// READOBJ-NEXT:      SymbolIndex: 12
// READOBJ-NEXT:    }
// READOBJ-NEXT:    Relocation {
// READOBJ-NEXT:      Offset: 0x20
// READOBJ-NEXT:      Type: IMAGE_REL_AMD64_REL32 (4)
// READOBJ-NEXT:      Symbol: g3
// READOBJ-NEXT:      SymbolIndex: 12
// READOBJ-NEXT:    }
// READOBJ-NEXT:    Relocation {
// READOBJ-NEXT:      Offset: 0x28
// READOBJ-NEXT:      Type: IMAGE_REL_AMD64_REL32 (4)
// READOBJ-NEXT:      Symbol: foobar
// READOBJ-NEXT:      SymbolIndex: 20
// READOBJ-NEXT:    }
// READOBJ-NEXT:    Relocation {
// READOBJ-NEXT:      Offset: 0x2C
// READOBJ-NEXT:      Type: IMAGE_REL_AMD64_REL32 (4)
// READOBJ-NEXT:      Symbol: foobar
// READOBJ-NEXT:      SymbolIndex: 20
// READOBJ-NEXT:    }
// READOBJ-NEXT:  }
// READOBJ-NEXT:]
