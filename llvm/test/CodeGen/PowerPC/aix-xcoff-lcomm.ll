; RUN: llc -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff < %s | FileCheck %s

; RUN: llc -mtriple powerpc-ibm-aix-xcoff -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --section-headers --file-header %t.o | \
; RUN:   FileCheck --check-prefixes=OBJ,OBJ32 %s
; RUN: llvm-readobj --syms %t.o | FileCheck --check-prefixes=SYMS,SYMS32 %s

; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -filetype=obj -o %t64.o < %s
; RUN: llvm-readobj --section-headers --file-header %t64.o | \
; RUN:   FileCheck --check-prefixes=OBJ,OBJ64 %s
; RUN: llvm-readobj --syms %t64.o | FileCheck --check-prefixes=SYMS,SYMS64 %s

@a = internal global i32 0, align 4
@b = internal global i64 0, align 8
@c = internal global i16 0, align 2

; CHECK:      .lcomm a,4,a[BS],2
; CHECK-NEXT: .lcomm b,8,b[BS],3
; CHECK-NEXT: .lcomm c,2,c[BS],1

; OBJ:        Arch: powerpc
; OBJ32-NEXT: AddressSize: 32bit
; OBJ64-NEXT: AddressSize: 64bit
; OBJ-NEXT:   FileHeader {
; OBJ32-NEXT:   Magic: 0x1DF
; OBJ64-NEXT:   Magic: 0x1F7
; OBJ-NEXT:     NumberOfSections: 2
; OBJ-NEXT:     TimeStamp:
; OBJ32-NEXT:   SymbolTableOffset: 0x64
; OBJ64-NEXT:   SymbolTableOffset: 0xA8
; OBJ-NEXT:     SymbolTableEntries: 9
; OBJ-NEXT:     OptionalHeaderSize: 0x0
; OBJ-NEXT:     Flags: 0x0
; OBJ-NEXT:   }
; OBJ-NEXT:   Sections [
; OBJ:          Section {{[{][[:space:]] *}}Index: 2
; OBJ-NEXT:       Name: .bss
; OBJ-NEXT:       PhysicalAddress: 0x0
; OBJ-NEXT:       VirtualAddress: 0x0
; OBJ-NEXT:       Size: 0x14
; OBJ-NEXT:       RawDataOffset: 0x0
; OBJ-NEXT:       RelocationPointer: 0x0
; OBJ-NEXT:       LineNumberPointer: 0x0
; OBJ-NEXT:       NumberOfRelocations: 0
; OBJ-NEXT:       NumberOfLineNumbers: 0
; OBJ-NEXT:       Type: STYP_BSS (0x80)
; OBJ-NEXT:     }
; OBJ-NEXT:   ]

; SYMS:      Symbols [
; SYMS:        Symbol {{[{][[:space:]] *}}Index: [[#Index:]]{{[[:space:]] *}}Name: a
; SYMS-NEXT:     Value (RelocatableAddress): 0x0
; SYMS-NEXT:     Section: .bss
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#Index + 1]]
; SYMS-NEXT:       SectionLen: 4
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 2
; SYMS-NEXT:       SymbolType: XTY_CM (0x3)
; SYMS-NEXT:       StorageMappingClass: XMC_BS (0x9)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }
; SYMS-NEXT:   Symbol {
; SYMS-NEXT:     Index: [[#Index + 2]]
; SYMS-NEXT:     Name: b
; SYMS-NEXT:     Value (RelocatableAddress): 0x8
; SYMS-NEXT:     Section: .bss
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#Index + 3]]
; SYMS-NEXT:       SectionLen: 8
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 3
; SYMS-NEXT:       SymbolType: XTY_CM (0x3)
; SYMS-NEXT:       StorageMappingClass: XMC_BS (0x9)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }
; SYMS-NEXT:   Symbol {
; SYMS-NEXT:     Index: [[#Index + 4]]
; SYMS-NEXT:     Name: c
; SYMS-NEXT:     Value (RelocatableAddress): 0x10
; SYMS-NEXT:     Section: .bss
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#Index + 5]]
; SYMS-NEXT:       SectionLen: 2
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 1
; SYMS-NEXT:       SymbolType: XTY_CM (0x3)
; SYMS-NEXT:       StorageMappingClass: XMC_BS (0x9)
; SYMS32-NEXT:     StabInfoIndex: 0x0
; SYMS32-NEXT:     StabSectNum: 0x0
; SYMS64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:     }
; SYMS-NEXT:   }
; SYMS-NEXT: ]
