; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc-ibm-aix-xcoff  < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc64-ibm-aix-xcoff < %s | FileCheck %s

; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc-ibm-aix-xcoff -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --section-headers --file-header %t.o | \
; RUN: FileCheck --check-prefix=OBJ %s
; RUN: llvm-readobj --syms %t.o | FileCheck --check-prefix=SYMS %s
; RUN: llvm-objdump -D %t.o | FileCheck --check-prefix=DIS %s

; RUN: not llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc64-ibm-aix-xcoff -filetype=obj < %s 2>&1 | \
; RUN: FileCheck --check-prefix=XCOFF64 %s
; XCOFF64: LLVM ERROR: 64-bit XCOFF object files are not supported yet.

@const_ivar = constant i32 35, align 4
@const_llvar = constant i64 36, align 8
@const_svar = constant i16 37, align 2
@const_fvar = constant float 8.000000e+02, align 4
@const_dvar = constant double 9.000000e+02, align 8
@const_over_aligned = constant double 9.000000e+02, align 32
@const_chrarray = constant [4 x i8] c"abcd", align 1
@const_dblarr = constant [4 x double] [double 1.000000e+00, double 2.000000e+00, double 3.000000e+00, double 4.000000e+00], align 8

; CHECK:               .csect .rodata[RO]
; CHECK-NEXT:          .globl  const_ivar
; CHECK-NEXT:          .align  2
; CHECK-NEXT:  const_ivar:
; CHECK-NEXT:          .long   35
; CHECK-NEXT:          .globl  const_llvar
; CHECK-NEXT:          .align  3
; CHECK-NEXT:  const_llvar:
; CHECK-NEXT:          .llong  36
; CHECK-NEXT:          .globl  const_svar
; CHECK-NEXT:          .align  1
; CHECK-NEXT:  const_svar:
; CHECK-NEXT:          .short  37
; CHECK-NEXT:          .globl  const_fvar
; CHECK-NEXT:          .align  2
; CHECK-NEXT:  const_fvar:
; CHECK-NEXT:          .long   1145569280
; CHECK-NEXT:          .globl  const_dvar
; CHECK-NEXT:          .align  3
; CHECK-NEXT:  const_dvar:
; CHECK-NEXT:          .llong  4651127699538968576
; CHECK-NEXT:          .globl  const_over_aligned
; CHECK-NEXT:          .align  5
; CHECK-NEXT:  const_over_aligned:
; CHECK-NEXT:          .llong  4651127699538968576
; CHECK-NEXT:          .globl  const_chrarray
; CHECK-NEXT:  const_chrarray:
; CHECK-NEXT:          .byte   97
; CHECK-NEXT:          .byte   98
; CHECK-NEXT:          .byte   99
; CHECK-NEXT:          .byte   100
; CHECK-NEXT:          .globl  const_dblarr
; CHECK-NEXT:          .align  3
; CHECK-NEXT:  const_dblarr:
; CHECK-NEXT:          .llong  4607182418800017408
; CHECK-NEXT:          .llong  4611686018427387904
; CHECK-NEXT:          .llong  4613937818241073152
; CHECK-NEXT:          .llong  4616189618054758400


; OBJ:      File: {{.*}}aix-xcoff-rodata.ll.tmp.o
; OBJ-NEXT: Format: aixcoff-rs6000
; OBJ-NEXT: Arch: powerpc
; OBJ-NEXT: AddressSize: 32bit
; OBJ-NEXT: FileHeader {
; OBJ-NEXT:   Magic: 0x1DF
; OBJ-NEXT:   NumberOfSections: 1
; OBJ-NEXT:   TimeStamp: None (0x0)
; OBJ-NEXT:   SymbolTableOffset: 0x8C
; OBJ-NEXT:   SymbolTableEntries: 20
; OBJ-NEXT:   OptionalHeaderSize: 0x0
; OBJ-NEXT:   Flags: 0x0
; OBJ-NEXT: }

; OBJ:      Sections [
; OBJ:        Section {
; OBJ-NEXT:     Index: 1
; OBJ-NEXT:     Name: .text
; OBJ-NEXT:     PhysicalAddress: 0x0
; OBJ-NEXT:     VirtualAddress: 0x0
; OBJ-NEXT:     Size: 0x50
; OBJ-NEXT:     RawDataOffset: 0x3C
; OBJ-NEXT:     RelocationPointer: 0x0
; OBJ-NEXT:     LineNumberPointer: 0x0
; OBJ-NEXT:     NumberOfRelocations: 0
; OBJ-NEXT:     NumberOfLineNumbers: 0
; OBJ-NEXT:     Type: STYP_TEXT (0x20)
; OBJ-NEXT:   }
; OBJ-NEXT: ]


; SYMS:       File: {{.*}}aix-xcoff-rodata.ll.tmp.o
; SYMS-NEXT:  Format: aixcoff-rs6000
; SYMS-NEXT:  Arch: powerpc
; SYMS-NEXT:  AddressSize: 32bit
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
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
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
; SYMS-NEXT:       ContainingCsectSymbolIndex: 2
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_RO (0x1)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
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
; SYMS-NEXT:       ContainingCsectSymbolIndex: 2
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_RO (0x1)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
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
; SYMS-NEXT:       ContainingCsectSymbolIndex: 2
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_RO (0x1)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
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
; SYMS-NEXT:       ContainingCsectSymbolIndex: 2
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_RO (0x1)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
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
; SYMS-NEXT:       ContainingCsectSymbolIndex: 2
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_RO (0x1)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
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
; SYMS-NEXT:       ContainingCsectSymbolIndex: 2
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_RO (0x1)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
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
; SYMS-NEXT:       ContainingCsectSymbolIndex: 2
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_RO (0x1)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
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
; SYMS-NEXT:       ContainingCsectSymbolIndex: 2
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_RO (0x1)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
; SYMS-NEXT:     }
; SYMS-NEXT:   }
; SYMS:      ]

; DIS: Disassembly of section .text:
; DIS: 00000000 const_ivar:
; DIS-NEXT:        0: 00 00 00 23
; DIS-NEXT:        4: 00 00 00 00

; DIS: 00000008 const_llvar:
; DIS-NEXT:        8: 00 00 00 00
; DIS-NEXT:        c: 00 00 00 24

; DIS: 00000010 const_svar:
; DIS-NEXT:       10: 00 25 00 00

; DIS: 00000014 const_fvar:
; DIS-NEXT:       14: 44 48 00 00

; DIS: 00000018 const_dvar:
; DIS-NEXT:       18: 40 8c 20 00
; DIS-NEXT:       1c: 00 00 00 00

; DIS: 00000020 const_over_aligned:
; DIS-NEXT:       20: 40 8c 20 00
; DIS-NEXT:       24: 00 00 00 00

; DIS: 00000028 const_chrarray:
; DIS-NEXT:       28: 61 62 63 64
; DIS-NEXT:       2c: 00 00 00 00

; DIS: 00000030 const_dblarr:
; DIS-NEXT:       30: 3f f0 00 00
; DIS-NEXT:       34: 00 00 00 00
; DIS-NEXT:       38: 40 00 00 00
; DIS-NEXT:       3c: 00 00 00 00
; DIS-NEXT:       40: 40 08 00 00
; DIS-NEXT:       44: 00 00 00 00
; DIS-NEXT:       48: 40 10 00 00
; DIS-NEXT:       4c: 00 00 00 00
