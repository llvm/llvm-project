; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN: -mattr=-altivec < %s | FileCheck --check-prefixes=COMMON,BIT32 %s

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr4 \
; RUN: -mattr=-altivec < %s | FileCheck --check-prefixes=COMMON,BIT64 %s

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN: -mattr=-altivec -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --symbols %t.o | FileCheck --check-prefix=CHECKSYM %s

; RUN: not --crash llc -verify-machineinstrs -mcpu=pwr4 -mtriple powerpc64-ibm-aix-xcoff \
; RUN: -mattr=-altivec -filetype=obj -o %t.o 2>&1 < %s | FileCheck --check-prefix=XCOFF64 %s
; XCOFF64: LLVM ERROR: 64-bit XCOFF object files are not supported yet.


@foo_ext_weak_p = global void (...)* bitcast (void ()* @foo_ext_weak_ref to void (...)*)
@b_w = extern_weak global i32

declare extern_weak void @foo_ext_weak_ref()

define i32 @main() {
entry:
  %0 = load void (...)*, void (...)** @foo_ext_weak_p
  %callee.knr.cast = bitcast void (...)* %0 to void ()*
  call void %callee.knr.cast()
  call void @foo_ext_weak(i32* @b_w)
  ret i32 0
}

declare extern_weak void @foo_ext_weak(i32*)

; COMMON:         .globl	main[DS]                # -- Begin function main
; COMMON-NEXT:    .globl	.main
; COMMON-NEXT:    .align	4
; COMMON-NEXT:    .csect main[DS]
; BIT32-NEXT:     .vbyte	4, .main                   # @main
; BIT32-NEXT:     .vbyte	4, TOC[TC0]
; BIT32-NEXT:     .vbyte	4, 0
; BIT64-NEXT:     .vbyte	8, .main                   # @main
; BIT64-NEXT:     .vbyte	8, TOC[TC0]
; BIT64-NEXT:     .vbyte	8, 0
; COMMON-NEXT:    .csect  .text[PR]
; COMMON-NEXT:    .main:

; COMMON:         .csect  .data[RW]
; COMMON:         .globl  foo_ext_weak_p
; BIT32-NEXT:     .align	2
; BIT64-NEXT:     .align 	3
; COMMON-NEXT: foo_ext_weak_p:
; BIT32-NEXT: 	  .vbyte	4, foo_ext_weak_ref[DS]
; BIT64-NEXT: 	  .vbyte	8, foo_ext_weak_ref[DS]
; COMMON-NEXT:    .weak   b_w[UA]
; COMMON-NEXT:    .weak   .foo_ext_weak_ref[PR]
; COMMON-NEXT:    .weak   foo_ext_weak_ref[DS]
; COMMON-NEXT:    .weak   .foo_ext_weak[PR]
; COMMON-NEXT:    .weak   foo_ext_weak[DS]
; COMMON-NEXT:    .toc
; COMMON-NEXT: L..C0:
; COMMON-NEXT:    .tc foo_ext_weak_p[TC],foo_ext_weak_p
; COMMON-NEXT: L..C1:
; COMMON-NEXT:    .tc b_w[TC],b_w[UA]

; CHECKSYM:      Symbols [
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index:]]
; CHECKSYM-NEXT:     Name: .foo_ext_weak
; CHECKSYM-NEXT:     Value (RelocatableAddress): 0x0
; CHECKSYM-NEXT:     Section: N_UNDEF
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_WEAKEXT (0x6F)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+1]]
; CHECKSYM-NEXT:       SectionLen: 0
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM-NEXT:       SymbolAlignmentLog2: 0
; CHECKSYM-NEXT:       SymbolType: XTY_ER (0x0)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_PR (0x0)
; CHECKSYM-NEXT:       StabInfoIndex: 0x0
; CHECKSYM-NEXT:       StabSectNum: 0x0
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index+2]]
; CHECKSYM-NEXT:     Name: foo_ext_weak_ref
; CHECKSYM-NEXT:     Value (RelocatableAddress): 0x0
; CHECKSYM-NEXT:     Section: N_UNDEF
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_WEAKEXT (0x6F)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+3]]
; CHECKSYM-NEXT:       SectionLen: 0
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM-NEXT:       SymbolAlignmentLog2: 0
; CHECKSYM-NEXT:       SymbolType: XTY_ER (0x0)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_DS (0xA)
; CHECKSYM-NEXT:       StabInfoIndex: 0x0
; CHECKSYM-NEXT:       StabSectNum: 0x0
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index+4]]
; CHECKSYM-NEXT:     Name: b_w
; CHECKSYM-NEXT:     Value (RelocatableAddress): 0x0
; CHECKSYM-NEXT:     Section: N_UNDEF
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_WEAKEXT (0x6F)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+5]]
; CHECKSYM-NEXT:       SectionLen: 0
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM-NEXT:       SymbolAlignmentLog2: 0
; CHECKSYM-NEXT:       SymbolType: XTY_ER (0x0)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_UA (0x4)
; CHECKSYM-NEXT:       StabInfoIndex: 0x0
; CHECKSYM-NEXT:       StabSectNum: 0x0
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index+6]]
; CHECKSYM-NEXT:     Name: .foo_ext_weak_ref
; CHECKSYM-NEXT:     Value (RelocatableAddress): 0x0
; CHECKSYM-NEXT:     Section: N_UNDEF
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_WEAKEXT (0x6F)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+7]]
; CHECKSYM-NEXT:       SectionLen: 0
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM-NEXT:       SymbolAlignmentLog2: 0
; CHECKSYM-NEXT:       SymbolType: XTY_ER (0x0)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_PR (0x0)
; CHECKSYM-NEXT:       StabInfoIndex: 0x0
; CHECKSYM-NEXT:       StabSectNum: 0x0
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index+8]]
; CHECKSYM-NEXT:     Name: foo_ext_weak
; CHECKSYM-NEXT:     Value (RelocatableAddress): 0x0
; CHECKSYM-NEXT:     Section: N_UNDEF
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_WEAKEXT (0x6F)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+9]]
; CHECKSYM-NEXT:       SectionLen: 0
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM-NEXT:       SymbolAlignmentLog2: 0
; CHECKSYM-NEXT:       SymbolType: XTY_ER (0x0)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_DS (0xA)
; CHECKSYM-NEXT:       StabInfoIndex: 0x0
; CHECKSYM-NEXT:       StabSectNum: 0x0
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
; CHECKSYM-NEXT:       SectionLen: 80
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM-NEXT:       SymbolAlignmentLog2: 4
; CHECKSYM-NEXT:       SymbolType: XTY_SD (0x1)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_PR (0x0)
; CHECKSYM-NEXT:       StabInfoIndex: 0x0
; CHECKSYM-NEXT:       StabSectNum: 0x0
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index+12]]
; CHECKSYM-NEXT:     Name: .main
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
; CHECKSYM-NEXT:       StabInfoIndex: 0x0
; CHECKSYM-NEXT:       StabSectNum: 0x0
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index+14]]
; CHECKSYM-NEXT:     Name: .data
; CHECKSYM-NEXT:     Value (RelocatableAddress): 0x50
; CHECKSYM-NEXT:     Section: .data
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+15]]
; CHECKSYM-NEXT:       SectionLen: 4
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM-NEXT:       SymbolAlignmentLog2: 2
; CHECKSYM-NEXT:       SymbolType: XTY_SD (0x1)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_RW (0x5)
; CHECKSYM-NEXT:       StabInfoIndex: 0x0
; CHECKSYM-NEXT:       StabSectNum: 0x0
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index+16]]
; CHECKSYM-NEXT:     Name: foo_ext_weak_p
; CHECKSYM-NEXT:     Value (RelocatableAddress): 0x50
; CHECKSYM-NEXT:     Section: .data
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_EXT (0x2)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+17]]
; CHECKSYM-NEXT:       ContainingCsectSymbolIndex: [[#Index+14]]
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM-NEXT:       SymbolAlignmentLog2: 0
; CHECKSYM-NEXT:       SymbolType: XTY_LD (0x2)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_RW (0x5)
; CHECKSYM-NEXT:       StabInfoIndex: 0x0
; CHECKSYM-NEXT:       StabSectNum: 0x0
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index+18]]
; CHECKSYM-NEXT:     Name: main
; CHECKSYM-NEXT:     Value (RelocatableAddress): 0x54
; CHECKSYM-NEXT:     Section: .data
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_EXT (0x2)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+19]]
; CHECKSYM-NEXT:       SectionLen: 12
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM-NEXT:       SymbolAlignmentLog2: 2
; CHECKSYM-NEXT:       SymbolType: XTY_SD (0x1)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_DS (0xA)
; CHECKSYM-NEXT:       StabInfoIndex: 0x0
; CHECKSYM-NEXT:       StabSectNum: 0x0
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index+20]]
; CHECKSYM-NEXT:     Name: TOC
; CHECKSYM-NEXT:     Value (RelocatableAddress): 0x60
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
; CHECKSYM-NEXT:       StabInfoIndex: 0x0
; CHECKSYM-NEXT:       StabSectNum: 0x0
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index+22]]
; CHECKSYM-NEXT:     Name: foo_ext_weak_p
; CHECKSYM-NEXT:     Value (RelocatableAddress): 0x60
; CHECKSYM-NEXT:     Section: .data
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+23]]
; CHECKSYM-NEXT:       SectionLen: 4
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM-NEXT:       SymbolAlignmentLog2: 2
; CHECKSYM-NEXT:       SymbolType: XTY_SD (0x1)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_TC (0x3)
; CHECKSYM-NEXT:       StabInfoIndex: 0x0
; CHECKSYM-NEXT:       StabSectNum: 0x0
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: [[#Index+24]]
; CHECKSYM-NEXT:     Name: b_w
; CHECKSYM-NEXT:     Value (RelocatableAddress): 0x64
; CHECKSYM-NEXT:     Section: .data
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: [[#Index+25]]
; CHECKSYM-NEXT:       SectionLen: 4
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM-NEXT:       SymbolAlignmentLog2: 2
; CHECKSYM-NEXT:       SymbolType: XTY_SD (0x1)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_TC (0x3)
; CHECKSYM-NEXT:       StabInfoIndex: 0x0
; CHECKSYM-NEXT:       StabSectNum: 0x0
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT: ]
