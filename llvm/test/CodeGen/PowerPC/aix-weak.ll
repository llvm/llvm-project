; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -xcoff-traceback-table=false -mcpu=pwr4 \
; RUN:   -mattr=-altivec -data-sections=false < %s | FileCheck --check-prefixes=COMMON,BIT32 %s

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -xcoff-traceback-table=false -mcpu=pwr4 \
; RUN:   -mattr=-altivec -data-sections=false < %s | FileCheck --check-prefixes=COMMON,BIT64 %s

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -xcoff-traceback-table=false -mcpu=pwr4 \
; RUN:   -mattr=-altivec -data-sections=false -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --symbols %t.o | FileCheck --check-prefixes=CHECKSYM,CHECKSYM32 %s

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -xcoff-traceback-table=false -mcpu=pwr4 \
; RUN:   -mattr=-altivec -data-sections=false -filetype=obj -o %t64.o < %s
; RUN: llvm-readobj --symbols %t64.o | FileCheck --check-prefixes=CHECKSYM,CHECKSYM64 %s

@foo_weak_p = global ptr @foo_ref_weak, align 4
@b = weak global i32 0, align 4

define weak void @foo_weak(ptr %p)  {
entry:
  %p.addr = alloca ptr, align 4
  store ptr %p, ptr %p.addr, align 4
  %0 = load ptr, ptr %p.addr, align 4
  %1 = load i32, ptr %0, align 4
  %inc = add nsw i32 %1, 1
  store i32 %inc, ptr %0, align 4
  ret void
}

define weak void @foo_ref_weak()  {
entry:
  ret void
}

define i32 @main()  {
entry:
  %0 = load ptr, ptr @foo_weak_p, align 4
  call void %0()
  call void @foo_weak(ptr @b)
  call void @foo_ref_weak()
  ret i32 0
}

; COMMON:               .weak	foo_weak[DS]            # -- Begin function foo_weak
; COMMON-NEXT:          .weak	.foo_weak
; COMMON-NEXT:          .align	4
; COMMON-NEXT:          .csect foo_weak[DS]
; BIT32-NEXT:           .vbyte	4, .foo_weak               # @foo_weak
; BIT32-NEXT:           .vbyte	4, TOC[TC0]
; BIT32-NEXT:           .vbyte	4, 0
; BIT64-NEXT:           .vbyte	8, .foo_weak               # @foo_weak
; BIT64-NEXT:           .vbyte	8, TOC[TC0]
; BIT64-NEXT:           .vbyte	8, 0
; COMMON-NEXT:          .csect .text[PR]
; COMMON-NEXT:  .foo_weak:

; COMMON:               .weak   foo_ref_weak[DS]        # -- Begin function foo_ref_weak
; COMMON-NEXT:          .weak	.foo_ref_weak
; COMMON-NEXT:          .align	4
; COMMON-NEXT:          .csect foo_ref_weak[DS]
; BIT32-NEXT:           .vbyte	4, .foo_ref_weak           # @foo_ref_weak
; BIT32-NEXT:           .vbyte	4, TOC[TC0]
; BIT32-NEXT:           .vbyte	4, 0
; BIT64-NEXT:           .vbyte	8, .foo_ref_weak           # @foo_ref_weak
; BIT64-NEXT:           .vbyte	8, TOC[TC0]
; BIT64-NEXT:           .vbyte	8, 0
; COMMON-NEXT:          .csect .text[PR]
; COMMON-NEXT:  .foo_ref_weak:

; COMMON:               .globl  main[DS]                # -- Begin function main
; COMMON-NEXT:          .globl  .main
; COMMON-NEXT:          .align	4
; COMMON-NEXT:          .csect main[DS]
; BIT32-NEXT:           .vbyte	4, .main                   # @main
; BIT32-NEXT:           .vbyte	4, TOC[TC0]
; BIT32-NEXT:           .vbyte	4, 0
; BIT64-NEXT:           .vbyte	8, .main                   # @main
; BIT64-NEXT:           .vbyte	8, TOC[TC0]
; BIT64-NEXT:           .vbyte	8, 0
; COMMON-NEXT:          .csect .text[PR]
; COMMON-NEXT:  .main:

; COMMON:     	        .csect .data[RW]
; COMMON-NEXT:          .globl	foo_weak_p
; BIT32-NEXT:           .align	2
; BIT64-NEXT:           .align	3
; COMMON-NEXT:  foo_weak_p:
; BIT32-NEXT:           .vbyte	4, foo_ref_weak[DS]
; BIT64-NEXT:           .vbyte	8, foo_ref_weak[DS]
; COMMON-NEXT:          .weak	b
; COMMON-NEXT:          .align	2
; COMMON-NEXT:  b:
; COMMON-NEXT:          .vbyte	4, 0                       # 0x0
; COMMON-NEXT:          .toc
; COMMON-NEXT:  L..C0:
; COMMON-NEXT:          .tc foo_weak_p[TC],foo_weak_p
; COMMON-NEXT:  L..C1:
; COMMON-NEXT:          .tc b[TC],b


; CHECKSYM:      Symbols [
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: 0
; CHECKSYM-NEXT:     Name: <stdin>
; CHECKSYM-NEXT:     Value (SymbolTableIndex): 0x0
; CHECKSYM-NEXT:     Section: N_DEBUG
; CHECKSYM-NEXT:     Source Language ID: TB_C (0x0)
; CHECKSYM-NEXT:     CPU Version ID: 0x0
; CHECKSYM-NEXT:     StorageClass: C_FILE (0x67)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 0
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index:]]
; CHECKSYM-NEXT:     Name: .text
; CHECKSYM-NEXT:     Value (RelocatableAddress): 0x0
; CHECKSYM-NEXT:     Section: .text
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+1]]
; CHECKSYM-NEXT:       SectionLen: 136
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM-NEXT:       SymbolAlignmentLog2: 5
; CHECKSYM-NEXT:       SymbolType: XTY_SD (0x1)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_PR (0x0)
; CHECKSYM32-NEXT:     StabInfoIndex: 0x0
; CHECKSYM32-NEXT:     StabSectNum: 0x0
; CHECKSYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index+2]]
; CHECKSYM-NEXT:     Name: .foo_weak
; CHECKSYM-NEXT:     Value (RelocatableAddress): 0x0
; CHECKSYM-NEXT:     Section: .text
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_WEAKEXT (0x6F)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+3]]
; CHECKSYM-NEXT:       ContainingCsectSymbolIndex: [[#Index]]
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM-NEXT:       SymbolAlignmentLog2: 0
; CHECKSYM-NEXT:       SymbolType: XTY_LD (0x2)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_PR (0x0)
; CHECKSYM32-NEXT:     StabInfoIndex: 0x0
; CHECKSYM32-NEXT:     StabSectNum: 0x0
; CHECKSYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index+4]]
; CHECKSYM-NEXT:     Name: .foo_ref_weak
; CHECKSYM-NEXT:     Value (RelocatableAddress): 0x20
; CHECKSYM-NEXT:     Section: .text
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_WEAKEXT (0x6F)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+5]]
; CHECKSYM-NEXT:       ContainingCsectSymbolIndex: [[#Index]]
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM-NEXT:       SymbolAlignmentLog2: 0
; CHECKSYM-NEXT:       SymbolType: XTY_LD (0x2)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_PR (0x0)
; CHECKSYM32-NEXT:     StabInfoIndex: 0x0
; CHECKSYM32-NEXT:     StabSectNum: 0x0
; CHECKSYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index+6]]
; CHECKSYM-NEXT:     Name: .main
; CHECKSYM-NEXT:     Value (RelocatableAddress): 0x30
; CHECKSYM-NEXT:     Section: .text
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_EXT (0x2)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+7]]
; CHECKSYM-NEXT:       ContainingCsectSymbolIndex: [[#Index]]
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM-NEXT:       SymbolAlignmentLog2: 0
; CHECKSYM-NEXT:       SymbolType: XTY_LD (0x2)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_PR (0x0)
; CHECKSYM32-NEXT:     StabInfoIndex: 0x0
; CHECKSYM32-NEXT:     StabSectNum: 0x0
; CHECKSYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index+8]]
; CHECKSYM-NEXT:     Name: .data
; CHECKSYM-NEXT:     Value (RelocatableAddress): 0x88
; CHECKSYM-NEXT:     Section: .data
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+9]]
; CHECKSYM32-NEXT:     SectionLen: 8
; CHECKSYM64-NEXT:     SectionLen: 12
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM32-NEXT:     SymbolAlignmentLog2: 2
; CHECKSYM64-NEXT:     SymbolAlignmentLog2: 3
; CHECKSYM-NEXT:       SymbolType: XTY_SD (0x1)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_RW (0x5)
; CHECKSYM32-NEXT:     StabInfoIndex: 0x0
; CHECKSYM32-NEXT:     StabSectNum: 0x0
; CHECKSYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index+10]]
; CHECKSYM-NEXT:     Name: foo_weak_p
; CHECKSYM-NEXT:     Value (RelocatableAddress): 0x88
; CHECKSYM-NEXT:     Section: .data
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_EXT (0x2)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+11]]
; CHECKSYM-NEXT:       ContainingCsectSymbolIndex: [[#Index+8]]
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM-NEXT:       SymbolAlignmentLog2: 0
; CHECKSYM-NEXT:       SymbolType: XTY_LD (0x2)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_RW (0x5)
; CHECKSYM32-NEXT:     StabInfoIndex: 0x0
; CHECKSYM32-NEXT:     StabSectNum: 0x0
; CHECKSYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index+12]]
; CHECKSYM-NEXT:     Name: b
; CHECKSYM32-NEXT:   Value (RelocatableAddress): 0x8C
; CHECKSYM64-NEXT:   Value (RelocatableAddress): 0x90
; CHECKSYM-NEXT:     Section: .data
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_WEAKEXT (0x6F)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+13]]
; CHECKSYM-NEXT:       ContainingCsectSymbolIndex: [[#Index+8]]
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM-NEXT:       SymbolAlignmentLog2: 0
; CHECKSYM-NEXT:       SymbolType: XTY_LD (0x2)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_RW (0x5)
; CHECKSYM32-NEXT:     StabInfoIndex: 0x0
; CHECKSYM32-NEXT:     StabSectNum: 0x0
; CHECKSYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index+14]]
; CHECKSYM-NEXT:     Name: foo_weak
; CHECKSYM32-NEXT:   Value (RelocatableAddress): 0x90
; CHECKSYM64-NEXT:   Value (RelocatableAddress): 0x98
; CHECKSYM-NEXT:     Section: .data
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_WEAKEXT (0x6F)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+15]]
; CHECKSYM32-NEXT:     SectionLen: 12
; CHECKSYM64-NEXT:     SectionLen: 24
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM32-NEXT:     SymbolAlignmentLog2: 2
; CHECKSYM64-NEXT:     SymbolAlignmentLog2: 3
; CHECKSYM-NEXT:       SymbolType: XTY_SD (0x1)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_DS (0xA)
; CHECKSYM32-NEXT:     StabInfoIndex: 0x0
; CHECKSYM32-NEXT:     StabSectNum: 0x0
; CHECKSYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index+16]]
; CHECKSYM-NEXT:     Name: foo_ref_weak
; CHECKSYM32-NEXT:   Value (RelocatableAddress): 0x9C
; CHECKSYM64-NEXT:   Value (RelocatableAddress): 0xB0
; CHECKSYM-NEXT:     Section: .data
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_WEAKEXT (0x6F)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+17]]
; CHECKSYM32-NEXT:     SectionLen: 12
; CHECKSYM64-NEXT:     SectionLen: 24
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM32-NEXT:     SymbolAlignmentLog2: 2
; CHECKSYM64-NEXT:     SymbolAlignmentLog2: 3
; CHECKSYM-NEXT:       SymbolType: XTY_SD (0x1)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_DS (0xA)
; CHECKSYM32-NEXT:     StabInfoIndex: 0x0
; CHECKSYM32-NEXT:     StabSectNum: 0x0
; CHECKSYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index+18]]
; CHECKSYM-NEXT:     Name: main
; CHECKSYM32-NEXT:   Value (RelocatableAddress): 0xA8
; CHECKSYM64-NEXT:   Value (RelocatableAddress): 0xC8
; CHECKSYM-NEXT:     Section: .data
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_EXT (0x2)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+19]]
; CHECKSYM32-NEXT:     SectionLen: 12
; CHECKSYM64-NEXT:     SectionLen: 24
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM32-NEXT:     SymbolAlignmentLog2: 2
; CHECKSYM64-NEXT:     SymbolAlignmentLog2: 3
; CHECKSYM-NEXT:       SymbolType: XTY_SD (0x1)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_DS (0xA)
; CHECKSYM32-NEXT:     StabInfoIndex: 0x0
; CHECKSYM32-NEXT:     StabSectNum: 0x0
; CHECKSYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index+20]]
; CHECKSYM-NEXT:     Name: TOC
; CHECKSYM32-NEXT:   Value (RelocatableAddress): 0xB4
; CHECKSYM64-NEXT:   Value (RelocatableAddress): 0xE0
; CHECKSYM-NEXT:     Section: .data
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+21]]
; CHECKSYM-NEXT:       SectionLen: 0
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM-NEXT:       SymbolAlignmentLog2: 2
; CHECKSYM-NEXT:       SymbolType: XTY_SD (0x1)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_TC0 (0xF)
; CHECKSYM32-NEXT:     StabInfoIndex: 0x0
; CHECKSYM32-NEXT:     StabSectNum: 0x0
; CHECKSYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index+22]]
; CHECKSYM-NEXT:     Name: foo_weak_p
; CHECKSYM32-NEXT:   Value (RelocatableAddress): 0xB4
; CHECKSYM64-NEXT:   Value (RelocatableAddress): 0xE0
; CHECKSYM-NEXT:     Section: .data
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+23]]
; CHECKSYM32-NEXT:     SectionLen: 4
; CHECKSYM64-NEXT:     SectionLen: 8
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM32-NEXT:     SymbolAlignmentLog2: 2
; CHECKSYM64-NEXT:     SymbolAlignmentLog2: 3
; CHECKSYM-NEXT:       SymbolType: XTY_SD (0x1)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_TC (0x3)
; CHECKSYM32-NEXT:     StabInfoIndex: 0x0
; CHECKSYM32-NEXT:     StabSectNum: 0x0
; CHECKSYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index+24]]
; CHECKSYM-NEXT:     Name: b
; CHECKSYM32-NEXT:   Value (RelocatableAddress): 0xB8
; CHECKSYM64-NEXT:   Value (RelocatableAddress): 0xE8
; CHECKSYM-NEXT:     Section: .data
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+25]]
; CHECKSYM32-NEXT:     SectionLen: 4
; CHECKSYM64-NEXT:     SectionLen: 8
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM32-NEXT:     SymbolAlignmentLog2: 2
; CHECKSYM64-NEXT:     SymbolAlignmentLog2: 3
; CHECKSYM-NEXT:       SymbolType: XTY_SD (0x1)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_TC (0x3)
; CHECKSYM32-NEXT:     StabInfoIndex: 0x0
; CHECKSYM32-NEXT:     StabSectNum: 0x0
; CHECKSYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT: ]
