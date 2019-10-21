; RUN: llc -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s

; RUN: llc -mtriple powerpc-ibm-aix-xcoff -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --section-headers --file-header %t.o | \
; RUN: FileCheck --check-prefix=OBJ %s
; RUN: llvm-readobj --syms %t.o | FileCheck --check-prefix=SYMS %s

; RUN: not llc -mtriple powerpc64-ibm-aix-xcoff -filetype=obj -o %t.o 2>&1 \
; RUN: < %s | FileCheck --check-prefix=XCOFF64 %s

; XCOFF64: LLVM ERROR: 64-bit XCOFF object files are not supported yet.

@a = common global i32 0, align 4
@b = common global i64 0, align 8
@c = common global i16 0, align 2

@d = common local_unnamed_addr global double 0.000000e+00, align 8
@f = common local_unnamed_addr global float 0.000000e+00, align 4

@over_aligned = common local_unnamed_addr global double 0.000000e+00, align 32

@array = common local_unnamed_addr global [33 x i8] zeroinitializer, align 1

; CHECK-NOT: .toc

; CHECK:      .csect .text[PR]
; CHECK-NEXT:  .file
; CHECK-NEXT: .comm   a,4,2
; CHECK-NEXT: .comm   b,8,3
; CHECK-NEXT: .comm   c,2,1
; CHECK-NEXT: .comm   d,8,3
; CHECK-NEXT: .comm   f,4,2
; CHECK-NEXT: .comm   over_aligned,8,5
; CHECK-NEXT: .comm   array,33,0

; OBJ:      File: {{.*}}aix-xcoff-common.ll.tmp.o
; OBJ-NEXT: Format: aixcoff-rs6000
; OBJ-NEXT: Arch: powerpc
; OBJ-NEXT: AddressSize: 32bit
; OBJ-NEXT: FileHeader {
; OBJ-NEXT:   Magic: 0x1DF
; OBJ-NEXT:   NumberOfSections: 2
; OBJ-NEXT:   TimeStamp:
; OBJ-NEXT:   SymbolTableOffset: 0x64
; OBJ-NEXT:   SymbolTableEntries: 16
; OBJ-NEXT:   OptionalHeaderSize: 0x0
; OBJ-NEXT:   Flags: 0x0
; OBJ-NEXT: }
; OBJ-NEXT: Sections [
; OBJ-NEXT:   Section {
; OBJ-NEXT:     Index: 1
; OBJ-NEXT:     Name: .text
; OBJ-NEXT:     PhysicalAddress: 0x0
; OBJ-NEXT:     VirtualAddress: 0x0
; OBJ-NEXT:     Size: 0x0
; OBJ-NEXT:     RawDataOffset: 0x64
; OBJ-NEXT:     RelocationPointer: 0x0
; OBJ-NEXT:     LineNumberPointer: 0x0
; OBJ-NEXT:     NumberOfRelocations: 0
; OBJ-NEXT:     NumberOfLineNumbers: 0
; OBJ-NEXT:     Type: STYP_TEXT (0x20)
; OBJ-NEXT:   }
; OBJ-NEXT:   Section {
; OBJ-NEXT:     Index: 2
; OBJ-NEXT:     Name: .bss
; OBJ-NEXT:     PhysicalAddress: 0x0
; OBJ-NEXT:     VirtualAddress: 0x0
; OBJ-NEXT:     Size: 0x6C
; OBJ-NEXT:     RawDataOffset: 0x0
; OBJ-NEXT:     RelocationPointer: 0x0
; OBJ-NEXT:     LineNumberPointer: 0x0
; OBJ-NEXT:     NumberOfRelocations: 0
; OBJ-NEXT:     NumberOfLineNumbers: 0
; OBJ-NEXT:     Type: STYP_BSS (0x80)
; OBJ-NEXT:   }
; OBJ-NEXT: ]

; SYMS:      File: {{.*}}aix-xcoff-common.ll.tmp.o
; SYMS-NEXT: Format: aixcoff-rs6000
; SYMS-NEXT: Arch: powerpc
; SYMS-NEXT: AddressSize: 32bit
; SYMS-NEXT: Symbols [
; SYMS:        Symbol {{[{][[:space:]] *}}Index: [[#Index:]]{{[[:space:]] *}}Name: a
; SYMS-NEXT:     Value (RelocatableAddress): 0x0
; SYMS-NEXT:     Section: .bss
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#Index + 1]]
; SYMS-NEXT:       SectionLen: 4
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 2
; SYMS-NEXT:       SymbolType: XTY_CM (0x3)
; SYMS-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
; SYMS-NEXT:     }
; SYMS-NEXT:   }
; SYMS-NEXT:   Symbol {
; SYMS-NEXT:     Index: [[#Index + 2]]
; SYMS-NEXT:     Name: b
; SYMS-NEXT:     Value (RelocatableAddress): 0x8
; SYMS-NEXT:     Section: .bss
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#Index + 3]]
; SYMS-NEXT:       SectionLen: 8
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 3
; SYMS-NEXT:       SymbolType: XTY_CM (0x3)
; SYMS-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
; SYMS-NEXT:     }
; SYMS-NEXT:   }
; SYMS-NEXT:   Symbol {
; SYMS-NEXT:     Index: [[#Index + 4]]
; SYMS-NEXT:     Name: c
; SYMS-NEXT:     Value (RelocatableAddress): 0x10
; SYMS-NEXT:     Section: .bss
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#Index + 5]]
; SYMS-NEXT:       SectionLen: 2
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 1
; SYMS-NEXT:       SymbolType: XTY_CM (0x3)
; SYMS-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
; SYMS-NEXT:     }
; SYMS-NEXT:   }
; SYMS-NEXT:   Symbol {
; SYMS-NEXT:     Index: [[#Index + 6]]
; SYMS-NEXT:     Name: d
; SYMS-NEXT:     Value (RelocatableAddress): 0x18
; SYMS-NEXT:     Section: .bss
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#Index + 7]]
; SYMS-NEXT:       SectionLen: 8
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 3
; SYMS-NEXT:       SymbolType: XTY_CM (0x3)
; SYMS-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
; SYMS-NEXT:     }
; SYMS-NEXT:   }
; SYMS-NEXT:   Symbol {
; SYMS-NEXT:     Index: [[#Index + 8]]
; SYMS-NEXT:     Name: f
; SYMS-NEXT:     Value (RelocatableAddress): 0x20
; SYMS-NEXT:     Section: .bss
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#Index + 9]]
; SYMS-NEXT:       SectionLen: 4
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 2
; SYMS-NEXT:       SymbolType: XTY_CM (0x3)
; SYMS-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
; SYMS-NEXT:     }
; SYMS-NEXT:   }
; SYMS-NEXT:   Symbol {
; SYMS-NEXT:     Index: [[#Index + 10]]
; SYMS-NEXT:     Name: over_aligned
; SYMS-NEXT:     Value (RelocatableAddress): 0x40
; SYMS-NEXT:     Section: .bss
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#Index + 11]]
; SYMS-NEXT:       SectionLen: 8
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 5
; SYMS-NEXT:       SymbolType: XTY_CM (0x3)
; SYMS-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
; SYMS-NEXT:     }
; SYMS-NEXT:   }
; SYMS-NEXT:   Symbol {
; SYMS-NEXT:     Index: [[#Index + 12]]
; SYMS-NEXT:     Name: array
; SYMS-NEXT:     Value (RelocatableAddress): 0x48
; SYMS-NEXT:     Section: .bss
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#Index + 13]]
; SYMS-NEXT:       SectionLen: 33
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_CM (0x3)
; SYMS-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
; SYMS-NEXT:     }
; SYMS-NEXT:   }
; SYMS-NEXT: ]
