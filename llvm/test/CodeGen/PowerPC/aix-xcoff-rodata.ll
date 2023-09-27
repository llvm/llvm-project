; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc-ibm-aix-xcoff -data-sections=false -ppc-merge-string-pool=false < %s | \
; RUN:   FileCheck --check-prefixes=CHECK,CHECK32 %s
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc64-ibm-aix-xcoff -data-sections=false -ppc-merge-string-pool=false < %s | \
; RUN:   FileCheck --check-prefixes=CHECK,CHECK64 %s

; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc-ibm-aix-xcoff -data-sections=false -ppc-merge-string-pool=false \
; RUN:   -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --section-headers --file-header %t.o | \
; RUN:   FileCheck --check-prefixes=OBJ,OBJ32 %s
; RUN: llvm-readobj --syms %t.o | FileCheck --check-prefixes=SYMS,SYMS32 %s
; RUN: llvm-objdump -D %t.o | FileCheck --check-prefix=DIS %s

; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc64-ibm-aix-xcoff -data-sections=false -ppc-merge-string-pool=false \
; RUN:   -filetype=obj -o %t64.o < %s
; RUN: llvm-readobj --section-headers --file-header %t64.o | \
; RUN:   FileCheck --check-prefixes=OBJ,OBJ64 %s
; RUN: llvm-readobj --syms %t64.o | FileCheck --check-prefixes=SYMS,SYMS64 %s

@const_ivar = constant i32 35, align 4
@const_llvar = constant i64 36, align 8
@const_svar = constant i16 37, align 2
@const_fvar = constant float 8.000000e+02, align 4
@const_dvar = constant double 9.000000e+02, align 8
@const_over_aligned = constant double 9.000000e+02, align 32
@const_chrarray = constant [4 x i8] c"abcd", align 1
@const_dblarr = constant [4 x double] [double 1.000000e+00, double 2.000000e+00, double 3.000000e+00, double 4.000000e+00], align 8

; CHECK:               .csect .rodata[RO],5
; CHECK-NEXT:          .globl  const_ivar
; CHECK-NEXT:          .align  2
; CHECK-NEXT:  const_ivar:
; CHECK-NEXT:          .vbyte	4, 35
; CHECK-NEXT:          .globl  const_llvar
; CHECK-NEXT:          .align  3
; CHECK-NEXT:  const_llvar:
; CHECK32-NEXT:        .vbyte	4, 0
; CHECK32-NEXT:        .vbyte	4, 36
; CHECK64-NEXT:        .vbyte	8, 36
; CHECK-NEXT:          .globl  const_svar
; CHECK-NEXT:          .align  1
; CHECK-NEXT:  const_svar:
; CHECK-NEXT:          .vbyte	2, 37
; CHECK-NEXT:          .globl  const_fvar
; CHECK-NEXT:          .align  2
; CHECK-NEXT:  const_fvar:
; CHECK-NEXT:          .vbyte	4, 0x44480000
; CHECK-NEXT:          .globl  const_dvar
; CHECK-NEXT:          .align  3
; CHECK-NEXT:  const_dvar:
; CHECK32-NEXT:        .vbyte	4, 1082925056
; CHECK32-NEXT:        .vbyte	4, 0
; CHECK64-NEXT:        .vbyte	8, 0x408c200000000000
; CHECK-NEXT:          .globl  const_over_aligned
; CHECK-NEXT:          .align  5
; CHECK-NEXT:  const_over_aligned:
; CHECK32-NEXT:        .vbyte	4, 1082925056
; CHECK32-NEXT:        .vbyte	4, 0
; CHECK64-NEXT:        .vbyte	8, 0x408c200000000000
; CHECK-NEXT:          .globl  const_chrarray
; CHECK-NEXT:  const_chrarray:
; CHECK-NEXT:          .byte   "abcd"
; CHECK-NEXT:          .globl  const_dblarr
; CHECK-NEXT:          .align  3
; CHECK-NEXT:  const_dblarr:
; CHECK32-NEXT:        .vbyte	4, 1072693248
; CHECK32-NEXT:        .vbyte	4, 0
; CHECK64-NEXT:        .vbyte	8, 0x3ff0000000000000
; CHECK32-NEXT:        .vbyte	4, 1073741824
; CHECK32-NEXT:        .vbyte	4, 0
; CHECK64-NEXT:        .vbyte	8, 0x4000000000000000
; CHECK32-NEXT:        .vbyte	4, 1074266112
; CHECK32-NEXT:        .vbyte	4, 0
; CHECK64-NEXT:        .vbyte	8, 0x4008000000000000
; CHECK32-NEXT:        .vbyte	4, 1074790400
; CHECK32-NEXT:        .vbyte	4, 0
; CHECK64-NEXT:        .vbyte	8, 0x4010000000000000

; OBJ:      FileHeader {
; OBJ32-NEXT: Magic: 0x1DF
; OBJ64-NEXT: Magic: 0x1F7
; OBJ-NEXT:   NumberOfSections: 1
; OBJ-NEXT:   TimeStamp: None (0x0)
; OBJ32-NEXT: SymbolTableOffset: 0x8C
; OBJ64-NEXT: SymbolTableOffset: 0xB0
; OBJ-NEXT:   SymbolTableEntries: 21
; OBJ-NEXT:   OptionalHeaderSize: 0x0
; OBJ-NEXT:   Flags: 0x0
; OBJ-NEXT: }

; OBJ:      Sections [
; OBJ:        Section {
; OBJ-NEXT:     Index: 1
; OBJ-NEXT:     Name: .text
; OBJ-NEXT:     PhysicalAddress: 0x0
; OBJ-NEXT:     VirtualAddress: 0x0
; OBJ32-NEXT:   Size: 0x50
; OBJ32-NEXT:   RawDataOffset: 0x3C
; OBJ64-NEXT:   Size: 0x50
; OBJ64-NEXT:   RawDataOffset: 0x60
; OBJ-NEXT:     RelocationPointer: 0x0
; OBJ-NEXT:     LineNumberPointer: 0x0
; OBJ-NEXT:     NumberOfRelocations: 0
; OBJ-NEXT:     NumberOfLineNumbers: 0
; OBJ-NEXT:     Type: STYP_TEXT (0x20)
; OBJ-NEXT:   }
; OBJ-NEXT: ]

; SYMS:       Symbols [
; SYMS:        Symbol {{[{][[:space:]] *}}Index: [[#INDX:]]{{[[:space:]] *}}Name: .rodata
; SYMS-NEXT:     Value (RelocatableAddress): 0x0
; SYMS-NEXT:     Section: .text
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+1]]
; SYMS-NEXT:       SectionLen: 80
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 5
; SYMS-NEXT:       SymbolType: XTY_SD (0x1)
; SYMS-NEXT:       StorageMappingClass: XMC_RO (0x1)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-NEXT:     Index: [[#INDX+2]]
; SYMS-NEXT:     Name: const_ivar
; SYMS-NEXT:     Value (RelocatableAddress): 0x0
; SYMS-NEXT:     Section: .text
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+3]]
; SYMS-NEXT:       ContainingCsectSymbolIndex: [[#INDX]]
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_RO (0x1)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-NEXT:     Index: [[#INDX+4]]
; SYMS-NEXT:     Name: const_llvar
; SYMS-NEXT:     Value (RelocatableAddress): 0x8
; SYMS-NEXT:     Section: .text
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+5]]
; SYMS-NEXT:       ContainingCsectSymbolIndex: [[#INDX]]
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_RO (0x1)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-NEXT:     Index: [[#INDX+6]]
; SYMS-NEXT:     Name: const_svar
; SYMS-NEXT:     Value (RelocatableAddress): 0x10
; SYMS-NEXT:     Section: .text
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+7]]
; SYMS-NEXT:       ContainingCsectSymbolIndex: [[#INDX]]
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_RO (0x1)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-NEXT:     Index: [[#INDX+8]]
; SYMS-NEXT:     Name: const_fvar
; SYMS-NEXT:     Value (RelocatableAddress): 0x14
; SYMS-NEXT:     Section: .text
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+9]]
; SYMS-NEXT:       ContainingCsectSymbolIndex: [[#INDX]]
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_RO (0x1)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-NEXT:     Index: [[#INDX+10]]
; SYMS-NEXT:     Name: const_dvar
; SYMS-NEXT:     Value (RelocatableAddress): 0x18
; SYMS-NEXT:     Section: .text
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+11]]
; SYMS-NEXT:       ContainingCsectSymbolIndex: [[#INDX]]
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_RO (0x1)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-NEXT:     Index: [[#INDX+12]]
; SYMS-NEXT:     Name: const_over_aligned
; SYMS-NEXT:     Value (RelocatableAddress): 0x20
; SYMS-NEXT:     Section: .text
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+13]]
; SYMS-NEXT:       ContainingCsectSymbolIndex: [[#INDX]]
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_RO (0x1)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-NEXT:     Index: [[#INDX+14]]
; SYMS-NEXT:     Name: const_chrarray
; SYMS-NEXT:     Value (RelocatableAddress): 0x28
; SYMS-NEXT:     Section: .text
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+15]]
; SYMS-NEXT:       ContainingCsectSymbolIndex: [[#INDX]]
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_RO (0x1)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-NEXT:     Index: [[#INDX+16]]
; SYMS-NEXT:     Name: const_dblarr
; SYMS-NEXT:     Value (RelocatableAddress): 0x30
; SYMS-NEXT:     Section: .text
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+17]]
; SYMS-NEXT:       ContainingCsectSymbolIndex: [[#INDX]]
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_RO (0x1)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }
; SYMS:      ]

; DIS: Disassembly of section .text:
; DIS: 00000000 <const_ivar>:
; DIS-NEXT:        0: 00 00 00 23
; DIS-NEXT:        4: 00 00 00 00

; DIS: 00000008 <const_llvar>:
; DIS-NEXT:        8: 00 00 00 00
; DIS-NEXT:        c: 00 00 00 24

; DIS: 00000010 <const_svar>:
; DIS-NEXT:       10: 00 25 00 00

; DIS: 00000014 <const_fvar>:
; DIS-NEXT:       14: 44 48 00 00

; DIS: 00000018 <const_dvar>:
; DIS-NEXT:       18: 40 8c 20 00
; DIS-NEXT:       1c: 00 00 00 00

; DIS: 00000020 <const_over_aligned>:
; DIS-NEXT:       20: 40 8c 20 00
; DIS-NEXT:       24: 00 00 00 00

; DIS: 00000028 <const_chrarray>:
; DIS-NEXT:       28: 61 62 63 64
; DIS-NEXT:       2c: 00 00 00 00

; DIS: 00000030 <const_dblarr>:
; DIS-NEXT:       30: 3f f0 00 00
; DIS-NEXT:       34: 00 00 00 00
; DIS-NEXT:       38: 40 00 00 00
; DIS-NEXT:       3c: 00 00 00 00
; DIS-NEXT:       40: 40 08 00 00
; DIS-NEXT:       44: 00 00 00 00
; DIS-NEXT:       48: 40 10 00 00
; DIS-NEXT:       4c: 00 00 00 00
