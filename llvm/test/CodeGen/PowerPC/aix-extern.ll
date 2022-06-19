; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:   -mattr=-altivec -data-sections=false -xcoff-traceback-table=false < %s | \
; RUN:   FileCheck --check-prefixes=COMMON,BIT32 %s

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:   -mattr=-altivec -data-sections=false -xcoff-traceback-table=false < %s | \
; RUN:   FileCheck --check-prefixes=COMMON,BIT64 %s

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:   -mattr=-altivec -data-sections=false -xcoff-traceback-table=false \
; RUN:   -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --symbols %t.o | FileCheck --check-prefixes=CHECKSYM,CHECKSYM32 %s

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:   -mattr=-altivec -data-sections=false -xcoff-traceback-table=false \
; RUN:   -filetype=obj -o %t64.o < %s
; RUN: llvm-readobj --symbols %t64.o | FileCheck --check-prefixes=CHECKSYM,CHECKSYM64 %s

@bar_p = global i32 (...)* @bar_ref, align 4
@b_e = external global i32, align 4

; Function Attrs: noinline nounwind optnone
define void @foo() {
entry:
  ret void
}

declare i32 @bar_ref(...)

; Function Attrs: noinline nounwind optnone
define i32 @main() {
entry:
  %call = call i32 @bar_extern(i32* @b_e)
  call void @foo()
  %0 = load i32 (...)*, i32 (...)** @bar_p, align 4
  %callee.knr.cast = bitcast i32 (...)* %0 to i32 ()*
  %call1 = call i32 %callee.knr.cast()
  %call2 = call i32 bitcast (i32 (...)* @bar_ref to i32 ()*)()
  ret i32 0
}

declare i32 @bar_extern(i32*)


; COMMON:           .globl	foo[DS]                 # -- Begin function foo
; COMMON-NEXT:	    .globl	.foo
; COMMON-NEXT:	    .align	4
; COMMON-NEXT:	    .csect foo[DS]
; BIT32-NEXT:       .vbyte	4, .foo                    # @foo
; BIT32-NEXT:       .vbyte	4, TOC[TC0]
; BIT32-NEXT:       .vbyte	4, 0
; BIT64-NEXT:       .vbyte	8, .foo                    # @foo
; BIT64-NEXT:       .vbyte	8, TOC[TC0]
; BIT64-NEXT:       .vbyte	8, 0
; COMMON-NEXT:      .csect .text[PR]
; COMMON-NEXT: .foo:

; COMMON:           .globl	main[DS]                # -- Begin function main
; COMMON-NEXT:      .globl	.main
; COMMON-NEXT:      .align	4
; COMMON-NEXT:      .csect main[DS]
; BIT32-NEXT:       .vbyte	4, .main                   # @main
; BIT32-NEXT:       .vbyte	4, TOC[TC0]
; BIT32-NEXT:       .vbyte	4, 0
; BIT64-NEXT:       .vbyte	8, .main                   # @main
; BIT64-NEXT:       .vbyte	8, TOC[TC0]
; BIT64-NEXT:       .vbyte	8, 0
; COMMON-NEXT:      .csect .text[PR]
; COMMON-NEXT: .main:

; COMMON:           .csect .data[RW]
; COMMON-NEXT:	    .globl	bar_p
; BIT32-NEXT:       .align	2
; BIT64-NEXT:       .align	3
; COMMON-NEXT: bar_p:
; BIT32-NEXT:       .vbyte	4, bar_ref[DS]
; BIT64-NEXT:       .vbyte	8, bar_ref[DS]
; COMMON-NEXT:	    .extern	b_e[UA]
; COMMON-NEXT:      .extern .bar_ref
; COMMON-NEXT:      .extern bar_ref[DS]
; COMMON-NEXT:	    .extern	.bar_extern
; COMMON-NEXT:      .extern     bar_extern[DS]
; COMMON-NEXT:	    .toc
; COMMON-NEXT: L..C0:
; COMMON-NEXT:      .tc b_e[TC],b_e[UA]
; COMMON-NEXT: L..C1:
; COMMON-NEXT:      .tc bar_p[TC],bar_p

; CHECKSYM:       Symbols [
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: 0
; CHECKSYM-NEXT:     Name: .file
; CHECKSYM-NEXT:     Value (SymbolTableIndex): 0x0
; CHECKSYM-NEXT:     Section: N_DEBUG
; CHECKSYM-NEXT:     Source Language ID: TB_C (0x0)
; CHECKSYM-NEXT:     CPU Version ID: 0x0
; CHECKSYM-NEXT:     StorageClass: C_FILE (0x67)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 0
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index:]]
; CHECKSYM-NEXT:     Name: .bar_extern
; CHECKSYM-NEXT:     Value (RelocatableAddress): 0x0
; CHECKSYM-NEXT:     Section: N_UNDEF
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_EXT (0x2)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+1]]
; CHECKSYM-NEXT:       SectionLen: 0
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM-NEXT:       SymbolAlignmentLog2: 0
; CHECKSYM-NEXT:       SymbolType: XTY_ER (0x0)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_PR (0x0)
; CHECKSYM32-NEXT:     StabInfoIndex: 0x0
; CHECKSYM32-NEXT:     StabSectNum: 0x0
; CHECKSYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index+2]]
; CHECKSYM-NEXT:     Name: .bar_ref
; CHECKSYM-NEXT:     Value (RelocatableAddress): 0x0
; CHECKSYM-NEXT:     Section: N_UNDEF
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_EXT (0x2)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+3]]
; CHECKSYM-NEXT:       SectionLen: 0
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM-NEXT:       SymbolAlignmentLog2: 0
; CHECKSYM-NEXT:       SymbolType: XTY_ER (0x0)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_PR (0x0)
; CHECKSYM32-NEXT:     StabInfoIndex: 0x0
; CHECKSYM32-NEXT:     StabSectNum: 0x0
; CHECKSYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index+4]]
; CHECKSYM-NEXT:     Name: bar_ref
; CHECKSYM-NEXT:     Value (RelocatableAddress): 0x0
; CHECKSYM-NEXT:     Section: N_UNDEF
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_EXT (0x2)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+5]]
; CHECKSYM-NEXT:       SectionLen: 0
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM-NEXT:       SymbolAlignmentLog2: 0
; CHECKSYM-NEXT:       SymbolType: XTY_ER (0x0)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_DS (0xA)
; CHECKSYM32-NEXT:     StabInfoIndex: 0x0
; CHECKSYM32-NEXT:     StabSectNum: 0x0
; CHECKSYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index+6]]
; CHECKSYM-NEXT:     Name: b_e
; CHECKSYM-NEXT:     Value (RelocatableAddress): 0x0
; CHECKSYM-NEXT:     Section: N_UNDEF
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_EXT (0x2)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+7]]
; CHECKSYM-NEXT:       SectionLen: 0
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM-NEXT:       SymbolAlignmentLog2: 0
; CHECKSYM-NEXT:       SymbolType: XTY_ER (0x0)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_UA (0x4)
; CHECKSYM32-NEXT:     StabInfoIndex: 0x0
; CHECKSYM32-NEXT:     StabSectNum: 0x0
; CHECKSYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index+8]]
; CHECKSYM-NEXT:     Name: bar_extern
; CHECKSYM-NEXT:     Value (RelocatableAddress): 0x0
; CHECKSYM-NEXT:     Section: N_UNDEF
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_EXT (0x2)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+9]]
; CHECKSYM-NEXT:       SectionLen: 0
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM-NEXT:       SymbolAlignmentLog2: 0
; CHECKSYM-NEXT:       SymbolType: XTY_ER (0x0)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_DS (0xA)
; CHECKSYM32-NEXT:     StabInfoIndex: 0x0
; CHECKSYM32-NEXT:     StabSectNum: 0x0
; CHECKSYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index+10]]
; CHECKSYM-NEXT:     Name: .text
; CHECKSYM-NEXT:     Value (RelocatableAddress): 0x0
; CHECKSYM-NEXT:     Section: .text
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+11]]
; CHECKSYM-NEXT:       SectionLen: 112
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM-NEXT:       SymbolAlignmentLog2: 4
; CHECKSYM-NEXT:       SymbolType: XTY_SD (0x1)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_PR (0x0)
; CHECKSYM32-NEXT:     StabInfoIndex: 0x0
; CHECKSYM32-NEXT:     StabSectNum: 0x0
; CHECKSYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index+12]]
; CHECKSYM-NEXT:     Name: .foo
; CHECKSYM-NEXT:     Value (RelocatableAddress): 0x0
; CHECKSYM-NEXT:     Section: .text
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_EXT (0x2)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+13]]
; CHECKSYM-NEXT:       ContainingCsectSymbolIndex: [[#Index+10]]
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
; CHECKSYM-NEXT:     Index: [[#Index+14]]
; CHECKSYM-NEXT:     Name: .main
; CHECKSYM-NEXT:     Value (RelocatableAddress): 0x10
; CHECKSYM-NEXT:     Section: .text
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_EXT (0x2)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+15]]
; CHECKSYM-NEXT:       ContainingCsectSymbolIndex: [[#Index+10]]
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
; CHECKSYM-NEXT:     Index: [[#Index+16]]
; CHECKSYM-NEXT:     Name: .data
; CHECKSYM-NEXT:     Value (RelocatableAddress): 0x70
; CHECKSYM-NEXT:     Section: .data
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+17]]
; CHECKSYM32-NEXT:     SectionLen: 4
; CHECKSYM64-NEXT:     SectionLen: 8
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
; CHECKSYM-NEXT:     Index: [[#Index+18]]
; CHECKSYM-NEXT:     Name: bar_p
; CHECKSYM-NEXT:     Value (RelocatableAddress): 0x70
; CHECKSYM-NEXT:     Section: .data
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_EXT (0x2)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+19]]
; CHECKSYM-NEXT:       ContainingCsectSymbolIndex: [[#Index+16]]
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
; CHECKSYM-NEXT:     Index: [[#Index+20]]
; CHECKSYM-NEXT:     Name: foo
; CHECKSYM32-NEXT:   Value (RelocatableAddress): 0x74
; CHECKSYM64-NEXT:   Value (RelocatableAddress): 0x78
; CHECKSYM-NEXT:     Section: .data
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_EXT (0x2)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+21]]
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
; CHECKSYM-NEXT:     Index: [[#Index+22]]
; CHECKSYM-NEXT:     Name: main
; CHECKSYM32-NEXT:   Value (RelocatableAddress): 0x80
; CHECKSYM64-NEXT:   Value (RelocatableAddress): 0x90
; CHECKSYM-NEXT:     Section: .data
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_EXT (0x2)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+23]]
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
; CHECKSYM-NEXT:     Index: [[#Index+24]]
; CHECKSYM-NEXT:     Name: TOC
; CHECKSYM32-NEXT:   Value (RelocatableAddress): 0x8C
; CHECKSYM64-NEXT:   Value (RelocatableAddress): 0xA8
; CHECKSYM-NEXT:     Section: .data
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+25]]
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
; CHECKSYM-NEXT:     Index: [[#Index+26]]
; CHECKSYM-NEXT:     Name: b_e
; CHECKSYM32-NEXT:   Value (RelocatableAddress): 0x8C
; CHECKSYM64-NEXT:   Value (RelocatableAddress): 0xA8
; CHECKSYM-NEXT:     Section: .data
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+27]]
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
; CHECKSYM-NEXT:     Index: [[#Index+28]]
; CHECKSYM-NEXT:     Name: bar_p
; CHECKSYM32-NEXT:   Value (RelocatableAddress): 0x90
; CHECKSYM64-NEXT:   Value (RelocatableAddress): 0xB0
; CHECKSYM-NEXT:     Section: .data
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+29]]
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
