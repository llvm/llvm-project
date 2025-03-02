; This file tests the codegen of initialized and common variables in AIX
; assembly and XCOFF object files.

; RUN: llc -mtriple powerpc-ibm-aix-xcoff -data-sections=false < %s | FileCheck --check-prefixes=CHECK,CHECK32 %s
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -data-sections=false < %s | FileCheck --check-prefixes=CHECK,CHECK64 %s

; RUN: llc -mtriple powerpc-ibm-aix-xcoff -data-sections=false -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --section-headers --file-header %t.o | \
; RUN:   FileCheck --check-prefix=OBJ %s
; RUN: llvm-readobj --syms %t.o | FileCheck --check-prefixes=SYMS,SYMS32 %s

; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -data-sections=false -filetype=obj -o %t64.o < %s
; RUN: llvm-readobj --section-headers --file-header %t64.o | \
; RUN:   FileCheck --check-prefix=OBJ64 %s
; RUN: llvm-readobj --syms %t64.o | FileCheck --check-prefixes=SYMS,SYMS64 %s

@ivar = local_unnamed_addr global i32 35, align 4
@llvar = local_unnamed_addr global i64 36, align 8
@svar = local_unnamed_addr global i16 37, align 2
@fvar = local_unnamed_addr global float 8.000000e+02, align 4
@dvar = local_unnamed_addr global double 9.000000e+02, align 8
@over_aligned = local_unnamed_addr global double 9.000000e+02, align 32
@chrarray = local_unnamed_addr global [4 x i8] c"abcd", align 1
@dblarr = local_unnamed_addr global [4 x double] [double 1.000000e+00, double 2.000000e+00, double 3.000000e+00, double 4.000000e+00], align 8
@d_0 = global double 0.000000e+00, align 8
@s_0 = global i16 0, align 2
@f_0 = global float 0.000000e+00, align 4

%struct.anon = type <{ i32, double }>
@astruct = global [1 x %struct.anon] [%struct.anon <{ i32 1, double 7.000000e+00 }>], align 1

%struct.anon2 = type { double, i32 }
@bstruct = global [1 x %struct.anon2] [%struct.anon2 { double 7.000000e+00 , i32 1}], align 8

@a = common global i32 0, align 4
@b = common global i64 0, align 8
@c = common global i16 0, align 2

@d = common local_unnamed_addr global double 0.000000e+00, align 8
@f = common local_unnamed_addr global float 0.000000e+00, align 4

@over_aligned_comm = common local_unnamed_addr global double 0.000000e+00, align 32

@array = common local_unnamed_addr global [33 x i8] zeroinitializer, align 1

; CHECK-NOT: .toc

; CHECK:      .file
; CHECK-NEXT: .csect ..text..[PR],5
; CHECK-NEXT: .rename ..text..[PR],""
; CHECK-NEXT: .machine "PWR7"

; CHECK:      .csect .data[RW],5
; CHECK-NEXT: .globl  ivar
; CHECK-NEXT: .align  2
; CHECK-NEXT: ivar:
; CHECK-NEXT: .vbyte	4, 35

; CHECK:      .globl  llvar
; CHECK-NEXT: .align  3
; CHECK-NEXT: llvar:
; CHECK32-NEXT: .vbyte	4, 0
; CHECK32-NEXT: .vbyte	4, 36
; CHECK64-NEXT: .vbyte	8, 36

; CHECK:      .globl  svar
; CHECK-NEXT: .align  1
; CHECK-NEXT: svar:
; CHECK-NEXT: .vbyte	2, 37

; CHECK:      .globl  fvar
; CHECK-NEXT: .align  2
; CHECK-NEXT: fvar:
; CHECK-NEXT: .vbyte	4, 0x44480000

; CHECK:      .globl  dvar
; CHECK-NEXT: .align  3
; CHECK-NEXT: dvar:
; CHECK32-NEXT: .vbyte	4, 1082925056
; CHECK32-NEXT: .vbyte	4, 0
; CHECK64-NEXT: .vbyte	8, 0x408c200000000000

; CHECK:      .globl  over_aligned
; CHECK-NEXT: .align  5
; CHECK-NEXT: over_aligned:
; CHECK32-NEXT: .vbyte	4, 1082925056
; CHECK32-NEXT: .vbyte	4, 0
; CHECK64-NEXT: .vbyte	8, 0x408c200000000000

; CHECK:      .globl  chrarray
; CHECK-NEXT: chrarray:
; CHECK-NEXT: .byte   "abcd"

; CHECK:      .globl  dblarr
; CHECK-NEXT: .align  3
; CHECK-NEXT: dblarr:
; CHECK32-NEXT: .vbyte	4, 1072693248
; CHECK32-NEXT: .vbyte	4, 0
; CHECK64-NEXT: .vbyte	8, 0x3ff0000000000000
; CHECK32-NEXT: .vbyte	4, 1073741824
; CHECK32-NEXT: .vbyte	4, 0
; CHECK64-NEXT: .vbyte	8, 0x4000000000000000
; CHECK32-NEXT: .vbyte	4, 1074266112
; CHECK32-NEXT: .vbyte	4, 0
; CHECK64-NEXT: .vbyte	8, 0x4008000000000000
; CHECK32-NEXT: .vbyte	4, 1074790400
; CHECK32-NEXT: .vbyte	4, 0
; CHECK64-NEXT: .vbyte	8, 0x4010000000000000

; CHECK:      .globl  d_0
; CHECK-NEXT: .align 3
; CHECK-NEXT: d_0:
; CHECK32-NEXT: .vbyte	4, 0
; CHECK32-NEXT: .vbyte	4, 0
; CHECK64-NEXT: .vbyte	8, 0

; CHECK:      .globl  s_0
; CHECK-NEXT: .align  1
; CHECK-NEXT: s_0:
; CHECK-NEXT: .vbyte	2, 0

; CHECK:      .globl f_0
; CHECK-NEXT: .align 2
; CHECK-NEXT: f_0:
; CHECK-NEXT: .vbyte	4, 0

; CHECK:            .globl astruct
; CHECK-NEXT:   astruct:
; CHECK-NEXT:       .vbyte	4, 1
; CHECK32-NEXT:     .vbyte	4, 1075576832
; CHECK32-NEXT:     .vbyte	4, 0
; CHECK64-NEXT:     .vbyte	8, 0x401c000000000000

; CHECK:            .globl bstruct
; CHECK-NEXT:       .align 3
; CHECK-NEXT:   bstruct:
; CHECK32-NEXT:     .vbyte	4, 1075576832
; CHECK32-NEXT:     .vbyte	4, 0
; CHECK64-NEXT:     .vbyte	8, 0x401c000000000000
; CHECK-NEXT:       .vbyte	4, 1
; CHECK-NEXT:       .space	4


; CHECK-NEXT: .comm   a[RW],4,2
; CHECK-NEXT: .comm   b[RW],8,3
; CHECK-NEXT: .comm   c[RW],2,1
; CHECK-NEXT: .comm   d[RW],8,3
; CHECK-NEXT: .comm   f[RW],4,2
; CHECK-NEXT: .comm   over_aligned_comm[RW],8,5
; CHECK-NEXT: .comm   array[RW],33,0

; OBJ:      File: 
; OBJ-NEXT: Format: aixcoff-rs6000
; OBJ-NEXT: Arch: powerpc
; OBJ-NEXT: AddressSize: 32bit
; OBJ-NEXT: FileHeader {
; OBJ-NEXT:   Magic: 0x1DF
; OBJ-NEXT:   NumberOfSections: 3
; OBJ-NEXT:   TimeStamp:
; OBJ-NEXT:   SymbolTableOffset: 0x10C
; OBJ-NEXT:   SymbolTableEntries: 47
; OBJ-NEXT:   OptionalHeaderSize: 0x0
; OBJ-NEXT:   Flags: 0x0
; OBJ-NEXT: }

; OBJ:      Sections [
; OBJ:        Section {
; OBJ-NEXT:     Index: [[#OBJ_INDX:]]
; OBJ-NEXT:     Name: .text
; OBJ-NEXT:     PhysicalAddress: 0x0
; OBJ-NEXT:     VirtualAddress: 0x0
; OBJ-NEXT:     Size: 0x0
; OBJ-NEXT:     RawDataOffset: 0x8C
; OBJ-NEXT:     RelocationPointer: 0x0
; OBJ-NEXT:     LineNumberPointer: 0x0
; OBJ-NEXT:     NumberOfRelocations: 0
; OBJ-NEXT:     NumberOfLineNumbers: 0
; OBJ-NEXT:     Type: STYP_TEXT (0x20)
; OBJ-NEXT:   }

; OBJ:        Section {
; OBJ-NEXT:     Index: [[#OBJ_INDX+1]]
; OBJ-NEXT:     Name: .data
; OBJ-NEXT:     PhysicalAddress: 0x0
; OBJ-NEXT:     VirtualAddress: 0x0
; OBJ-NEXT:     Size: 0x80
; OBJ-NEXT:     RawDataOffset: 0x8C
; OBJ-NEXT:     RelocationPointer: 0x0
; OBJ-NEXT:     LineNumberPointer: 0x0
; OBJ-NEXT:     NumberOfRelocations: 0
; OBJ-NEXT:     NumberOfLineNumbers: 0
; OBJ-NEXT:     Type: STYP_DATA (0x40)
; OBJ-NEXT:   }

; OBJ:        Section {
; OBJ-NEXT:     Index: [[#OBJ_INDX+2]]
; OBJ-NEXT:     Name: .bss
; OBJ-NEXT:     PhysicalAddress: 0x80
; OBJ-NEXT:     VirtualAddress: 0x80
; OBJ-NEXT:     Size: 0x6C
; OBJ-NEXT:     RawDataOffset: 0x0
; OBJ-NEXT:     RelocationPointer: 0x0
; OBJ-NEXT:     LineNumberPointer: 0x0
; OBJ-NEXT:     NumberOfRelocations: 0
; OBJ-NEXT:     NumberOfLineNumbers: 0
; OBJ-NEXT:     Type: STYP_BSS (0x80)
; OBJ-NEXT:   }
; OBJ:      ]

; SYMS:      Symbols [
; SYMS-NEXT:   Symbol {
; SYMS-NEXT:     Index: 0
; SYMS-NEXT:     Name: .file
; SYMS-NEXT:     Value (SymbolTableIndex): 0x0
; SYMS-NEXT:     Section: N_DEBUG
; SYMS-NEXT:     Source Language ID: TB_CPLUSPLUS (0x9)
; SYMS-NEXT:     CPU Version ID: TCPU_PWR7 (0x18)
; SYMS-NEXT:     StorageClass: C_FILE (0x67)
; SYMS-NEXT:     NumberOfAuxEntries: 2
; SYMS-NEXT:     File Auxiliary Entry {
; SYMS-NEXT:       Index: 1
; SYMS-NEXT:       Name:
; SYMS-NEXT:       Type: XFT_FN (0x0)
; SYMS64-NEXT:     Auxiliary Type: AUX_FILE (0xFC)
; SYMS-NEXT:     }
; SYMS-NEXT:     File Auxiliary Entry {
; SYMS-NEXT:       Index: 2
; SYMS-NEXT:       Name: LLVM
; SYMS-NEXT:       Type: XFT_CV (0x2)
; SYMS64-NEXT:     Auxiliary Type: AUX_FILE (0xFC)
; SYMS-NEXT:     }
; SYMS-NEXT:   }
; SYMS-NEXT:   Symbol {
; SYMS-NEXT:     Index: [[#INDX:]]
; SYMS-NEXT:     Name:
; SYMS-NEXT:     Value (RelocatableAddress): 0x0
; SYMS-NEXT:     Section: .text
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+1]]
; SYMS-NEXT:       SectionLen: 0
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 5
; SYMS-NEXT:       SymbolType: XTY_SD (0x1)
; SYMS-NEXT:       StorageMappingClass: XMC_PR (0x0)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-NEXT:     Index: [[#INDX+2]]
; SYMS-NEXT:     Name: .data
; SYMS-NEXT:     Value (RelocatableAddress): 0x0
; SYMS-NEXT:     Section: .data
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+3]]
; SYMS-NEXT:       SectionLen: 128
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 5
; SYMS-NEXT:       SymbolType: XTY_SD (0x1)
; SYMS-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-NEXT:     Index: [[#INDX+4]]
; SYMS-NEXT:     Name: ivar
; SYMS-NEXT:     Value (RelocatableAddress): 0x0
; SYMS-NEXT:     Section: .data
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+5]]
; SYMS-NEXT:       ContainingCsectSymbolIndex: [[#INDX+2]]
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-NEXT:     Index: [[#INDX+6]]
; SYMS-NEXT:     Name: llvar
; SYMS-NEXT:     Value (RelocatableAddress): 0x8
; SYMS-NEXT:     Section: .data
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+7]]
; SYMS-NEXT:       ContainingCsectSymbolIndex: [[#INDX+2]]
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-NEXT:     Index: [[#INDX+8]]
; SYMS-NEXT:     Name: svar
; SYMS-NEXT:     Value (RelocatableAddress): 0x10
; SYMS-NEXT:     Section: .data
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+9]]
; SYMS-NEXT:       ContainingCsectSymbolIndex: [[#INDX+2]]
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-NEXT:     Index: [[#INDX+10]]
; SYMS-NEXT:     Name: fvar
; SYMS-NEXT:     Value (RelocatableAddress): 0x14
; SYMS-NEXT:     Section: .data
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+11]]
; SYMS-NEXT:       ContainingCsectSymbolIndex: [[#INDX+2]]
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-NEXT:     Index: [[#INDX+12]]
; SYMS-NEXT:     Name: dvar
; SYMS-NEXT:     Value (RelocatableAddress): 0x18
; SYMS-NEXT:     Section: .data
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+13]]
; SYMS-NEXT:       ContainingCsectSymbolIndex: [[#INDX+2]]
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-NEXT:     Index: [[#INDX+14]]
; SYMS-NEXT:     Name: over_aligned
; SYMS-NEXT:     Value (RelocatableAddress): 0x20
; SYMS-NEXT:     Section: .data
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+15]]
; SYMS-NEXT:       ContainingCsectSymbolIndex: [[#INDX+2]]
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-NEXT:     Index: [[#INDX+16]]
; SYMS-NEXT:     Name: chrarray
; SYMS-NEXT:     Value (RelocatableAddress): 0x28
; SYMS-NEXT:     Section: .data
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+17]]
; SYMS-NEXT:       ContainingCsectSymbolIndex: [[#INDX+2]]
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-NEXT:     Index: [[#INDX+18]]
; SYMS-NEXT:     Name: dblarr
; SYMS-NEXT:     Value (RelocatableAddress): 0x30
; SYMS-NEXT:     Section: .data
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+19]]
; SYMS-NEXT:       ContainingCsectSymbolIndex: [[#INDX+2]]
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-NEXT:     Index: [[#INDX+20]]
; SYMS-NEXT:     Name: d_0
; SYMS-NEXT:     Value (RelocatableAddress): 0x50
; SYMS-NEXT:     Section: .data
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+21]]
; SYMS-NEXT:       ContainingCsectSymbolIndex: [[#INDX+2]]
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-NEXT:     Index: [[#INDX+22]]
; SYMS-NEXT:     Name: s_0
; SYMS-NEXT:     Value (RelocatableAddress): 0x58
; SYMS-NEXT:     Section: .data
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+23]]
; SYMS-NEXT:       ContainingCsectSymbolIndex: [[#INDX+2]]
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-NEXT:     Index: [[#INDX+24]]
; SYMS-NEXT:     Name: f_0
; SYMS-NEXT:     Value (RelocatableAddress): 0x5C
; SYMS-NEXT:     Section: .data
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+25]]
; SYMS-NEXT:       ContainingCsectSymbolIndex: [[#INDX+2]]
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-NEXT:     Index: [[#INDX+26]]
; SYMS-NEXT:     Name: astruct
; SYMS-NEXT:     Value (RelocatableAddress): 0x60
; SYMS-NEXT:     Section: .data
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+27]]
; SYMS-NEXT:       ContainingCsectSymbolIndex: [[#INDX+2]]
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-NEXT:     Index: [[#INDX+28]]
; SYMS-NEXT:     Name: bstruct
; SYMS-NEXT:     Value (RelocatableAddress): 0x70
; SYMS-NEXT:     Section: .data
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+29]]
; SYMS-NEXT:       ContainingCsectSymbolIndex: [[#INDX+2]]
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-NEXT:     Index: [[#INDX+30]]
; SYMS-NEXT:     Name: a
; SYMS-NEXT:     Value (RelocatableAddress): 0x80
; SYMS-NEXT:     Section: .bss
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+31]]
; SYMS-NEXT:       SectionLen: 4
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 2
; SYMS-NEXT:       SymbolType: XTY_CM (0x3)
; SYMS-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-NEXT:     Index: [[#INDX+32]]
; SYMS-NEXT:     Name: b
; SYMS-NEXT:     Value (RelocatableAddress): 0x88
; SYMS-NEXT:     Section: .bss
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+33]]
; SYMS-NEXT:       SectionLen: 8
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 3
; SYMS-NEXT:       SymbolType: XTY_CM (0x3)
; SYMS-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-NEXT:     Index: [[#INDX+34]]
; SYMS-NEXT:     Name: c
; SYMS-NEXT:     Value (RelocatableAddress): 0x90
; SYMS-NEXT:     Section: .bss
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+35]]
; SYMS-NEXT:       SectionLen: 2
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 1
; SYMS-NEXT:       SymbolType: XTY_CM (0x3)
; SYMS-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-NEXT:     Index: [[#INDX+36]]
; SYMS-NEXT:     Name: d
; SYMS-NEXT:     Value (RelocatableAddress): 0x98
; SYMS-NEXT:     Section: .bss
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+37]]
; SYMS-NEXT:       SectionLen: 8
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 3
; SYMS-NEXT:       SymbolType: XTY_CM (0x3)
; SYMS-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-NEXT:     Index: [[#INDX+38]]
; SYMS-NEXT:     Name: f
; SYMS-NEXT:     Value (RelocatableAddress): 0xA0
; SYMS-NEXT:     Section: .bss
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+39]]
; SYMS-NEXT:       SectionLen: 4
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 2
; SYMS-NEXT:       SymbolType: XTY_CM (0x3)
; SYMS-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-NEXT:     Index: [[#INDX+40]]
; SYMS-NEXT:     Name: over_aligned_comm
; SYMS-NEXT:     Value (RelocatableAddress): 0xC0
; SYMS-NEXT:     Section: .bss
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+41]]
; SYMS-NEXT:       SectionLen: 8
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 5
; SYMS-NEXT:       SymbolType: XTY_CM (0x3)
; SYMS-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-NEXT:     Index: [[#INDX+42]]
; SYMS-NEXT:     Name: array
; SYMS-NEXT:     Value (RelocatableAddress): 0xC8
; SYMS-NEXT:     Section: .bss
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+43]]
; SYMS-NEXT:       SectionLen: 33
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_CM (0x3)
; SYMS-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }
; SYMS:      ]

; OBJ64:      Format: aix5coff64-rs6000
; OBJ64-NEXT: Arch: powerpc64
; OBJ64-NEXT: AddressSize: 64bit
; OBJ64-NEXT: FileHeader {
; OBJ64-NEXT:   Magic: 0x1F7
; OBJ64-NEXT:   NumberOfSections: 3
; OBJ64-NEXT:   TimeStamp: None (0x0)
; OBJ64-NEXT:   SymbolTableOffset: 0x170
; OBJ64-NEXT:   SymbolTableEntries: 47
; OBJ64-NEXT:   OptionalHeaderSize: 0x0
; OBJ64-NEXT:   Flags: 0x0
; OBJ64-NEXT: }

; OBJ64:      Sections [
; OBJ64-NEXT:   Section {
; OBJ64-NEXT:     Index: [[#OBJ64_INDX:]]
; OBJ64-NEXT:     Name: .text
; OBJ64-NEXT:     PhysicalAddress: 0x0
; OBJ64-NEXT:     VirtualAddress: 0x0
; OBJ64-NEXT:     Size: 0x0
; OBJ64-NEXT:     RawDataOffset: 0xF0
; OBJ64-NEXT:     RelocationPointer: 0x0
; OBJ64-NEXT:     LineNumberPointer: 0x0
; OBJ64-NEXT:     NumberOfRelocations: 0
; OBJ64-NEXT:     NumberOfLineNumbers: 0
; OBJ64-NEXT:     Type: STYP_TEXT (0x20)
; OBJ64-NEXT:   }
; OBJ64-NEXT:   Section {
; OBJ64-NEXT:     Index: [[#OBJ64_INDX+1]]
; OBJ64-NEXT:     Name: .data
; OBJ64-NEXT:     PhysicalAddress: 0x0
; OBJ64-NEXT:     VirtualAddress: 0x0
; OBJ64-NEXT:     Size: 0x80
; OBJ64-NEXT:     RawDataOffset: 0xF0
; OBJ64-NEXT:     RelocationPointer: 0x0
; OBJ64-NEXT:     LineNumberPointer: 0x0
; OBJ64-NEXT:     NumberOfRelocations: 0
; OBJ64-NEXT:     NumberOfLineNumbers: 0
; OBJ64-NEXT:     Type: STYP_DATA (0x40)
; OBJ64-NEXT:   }
; OBJ64-NEXT:   Section {
; OBJ64-NEXT:     Index: [[#OBJ64_INDX+2]]
; OBJ64-NEXT:     Name: .bss
; OBJ64-NEXT:     PhysicalAddress: 0x80
; OBJ64-NEXT:     VirtualAddress: 0x80
; OBJ64-NEXT:     Size: 0x6C
; OBJ64-NEXT:     RawDataOffset: 0x0
; OBJ64-NEXT:     RelocationPointer: 0x0
; OBJ64-NEXT:     LineNumberPointer: 0x0
; OBJ64-NEXT:     NumberOfRelocations: 0
; OBJ64-NEXT:     NumberOfLineNumbers: 0
; OBJ64-NEXT:     Type: STYP_BSS (0x80)
; OBJ64-NEXT:   }
; OBJ64-NEXT: ]
