; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false -data-sections=false -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --relocs --expand-relocs %t.o | FileCheck --check-prefix=RELOC %s
; RUN: llvm-readobj --syms %t.o | FileCheck --check-prefix=SYM %s
; RUN: llvm-objdump -D -r --symbol-description %t.o | FileCheck --check-prefix=DIS %s

@const_ivar = constant i32 6, align 4
@GInit = global i32 1, align 4
@TGInit = thread_local global i32 1, align 4
; @TIUninit is local-dynamic indeed
@TIUninit = internal thread_local global i32 0, align 4

; Function Attrs: nofree norecurse nounwind willreturn writeonly
define void @storesTIUninit(i32 %Val) #0 {
entry:
  store i32 %Val, ptr @TIUninit, align 4
  ret void
}

; Function Attrs: norecurse nounwind readonly willreturn
define i32 @loadsTGInit() #1 {
entry:
  %0 = load i32, ptr @TGInit, align 4
  %1 = load i32, ptr @GInit, align 4
  %add = add nsw i32 %1, %0
  ret i32 %add
}

; RELOC:      File: {{.*}}aix-tls-xcoff-reloc.ll.tmp.o
; RELOC-NEXT: Format: aixcoff-rs6000
; RELOC-NEXT: Arch: powerpc
; RELOC-NEXT: AddressSize: 32bit
; RELOC-NEXT: Relocations [
; RELOC-NEXT:   Section (index: 1) .text {
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x12
; RELOC-NEXT:     Symbol: TIUninit (25)
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 16
; RELOC-NEXT:     Type: R_TOC (0x3)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x16
; RELOC-NEXT:     Symbol: _$TLSML (27)
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 16
; RELOC-NEXT:     Type: R_TOC (0x3)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x18
; RELOC-NEXT:     Symbol: .__tls_get_mod (1)
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 26
; RELOC-NEXT:     Type: R_RBA (0x18)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x4A
; RELOC-NEXT:     Symbol: .TGInit (29)
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 16
; RELOC-NEXT:     Type: R_TOC (0x3)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x4E
; RELOC-NEXT:     Symbol: TGInit (31)
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 16
; RELOC-NEXT:     Type: R_TOC (0x3)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x54
; RELOC-NEXT:     Symbol: .__tls_get_addr (3)
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 26
; RELOC-NEXT:     Type: R_RBA (0x18)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x5A
; RELOC-NEXT:     Symbol: GInit (33)
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 16
; RELOC-NEXT:     Type: R_TOC (0x3)
; RELOC-NEXT:   }
; RELOC-NEXT: }
; RELOC-NEXT: Section (index: 2) .data {
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0x80
; RELOC-NEXT:   Symbol: .storesTIUninit (7)
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_POS (0x0)
; RELOC-NEXT: }
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0x84
; RELOC-NEXT:   Symbol: TOC (23)
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_POS (0x0)
; RELOC-NEXT: }
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0x8C
; RELOC-NEXT:   Symbol: .loadsTGInit (9)
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_POS (0x0)
; RELOC-NEXT: }
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0x90
; RELOC-NEXT:   Symbol: TOC (23)
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_POS (0x0)
; RELOC-NEXT: }
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0x98
; RELOC-NEXT:   Symbol: TIUninit (39)
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_TLS_LD (0x22)
; RELOC-NEXT: }
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0x9C
; RELOC-NEXT:   Symbol: _$TLSML (27)
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_TLSML (0x25)
; RELOC-NEXT: }
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0xA0
; RELOC-NEXT:   Symbol: TGInit (37)
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_TLSM (0x24)
; RELOC-NEXT: }
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0xA4
; RELOC-NEXT:   Symbol: TGInit (37)
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_TLS (0x20)
; RELOC-NEXT: }
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0xA8
; RELOC-NEXT:   Symbol: GInit (17)
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_POS (0x0)
; RELOC-NEXT: }
; RELOC-NEXT: }
; RELOC-NEXT: ]

; SYM:      File: {{.*}}aix-tls-xcoff-reloc.ll.tmp.o
; SYM-NEXT: Format: aixcoff-rs6000
; SYM-NEXT: Arch: powerpc
; SYM-NEXT: AddressSize: 32bit
; SYM-NEXT: Symbols [
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 0
; SYM-NEXT:     Name: <stdin>
; SYM-NEXT:     Value (SymbolTableIndex): 0x0
; SYM-NEXT:     Section: N_DEBUG
; SYM-NEXT:     Source Language ID: TB_CPLUSPLUS (0x9)
; SYM-NEXT:     CPU Version ID: TCPU_COM (0x3)
; SYM-NEXT:     StorageClass: C_FILE (0x67)
; SYM-NEXT:     NumberOfAuxEntries: 0
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 1
; SYM-NEXT:     Name: .__tls_get_mod
; SYM-NEXT:     Value (RelocatableAddress): 0x0
; SYM-NEXT:     Section: N_UNDEF
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 2
; SYM-NEXT:       SectionLen: 0
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 0
; SYM-NEXT:       SymbolType: XTY_ER (0x0)
; SYM-NEXT:       StorageMappingClass: XMC_PR (0x0)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 3
; SYM-NEXT:     Name: .__tls_get_addr
; SYM-NEXT:     Value (RelocatableAddress): 0x0
; SYM-NEXT:     Section: N_UNDEF
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 4
; SYM-NEXT:       SectionLen: 0
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 0
; SYM-NEXT:       SymbolType: XTY_ER (0x0)
; SYM-NEXT:       StorageMappingClass: XMC_PR (0x0)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 5
; SYM-NEXT:     Name: 
; SYM-NEXT:     Value (RelocatableAddress): 0x0
; SYM-NEXT:     Section: .text
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 6
; SYM-NEXT:       SectionLen: 120
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 5
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_PR (0x0)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 7
; SYM-NEXT:     Name: .storesTIUninit
; SYM-NEXT:     Value (RelocatableAddress): 0x0
; SYM-NEXT:     Section: .text
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 8
; SYM-NEXT:       ContainingCsectSymbolIndex: 5
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 0
; SYM-NEXT:       SymbolType: XTY_LD (0x2)
; SYM-NEXT:       StorageMappingClass: XMC_PR (0x0)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 9
; SYM-NEXT:     Name: .loadsTGInit
; SYM-NEXT:     Value (RelocatableAddress): 0x40
; SYM-NEXT:     Section: .text
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 10
; SYM-NEXT:       ContainingCsectSymbolIndex: 5
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 0
; SYM-NEXT:       SymbolType: XTY_LD (0x2)
; SYM-NEXT:       StorageMappingClass: XMC_PR (0x0)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 11
; SYM-NEXT:     Name: .rodata
; SYM-NEXT:     Value (RelocatableAddress): 0x78
; SYM-NEXT:     Section: .text
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 12
; SYM-NEXT:       SectionLen: 4
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 2
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_RO (0x1)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 13
; SYM-NEXT:     Name: const_ivar
; SYM-NEXT:     Value (RelocatableAddress): 0x78
; SYM-NEXT:     Section: .text
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 14
; SYM-NEXT:       ContainingCsectSymbolIndex: 11
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 0
; SYM-NEXT:       SymbolType: XTY_LD (0x2)
; SYM-NEXT:       StorageMappingClass: XMC_RO (0x1)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 15
; SYM-NEXT:     Name: .data
; SYM-NEXT:     Value (RelocatableAddress): 0x7C
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 16
; SYM-NEXT:       SectionLen: 4
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 2
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 17
; SYM-NEXT:     Name: GInit
; SYM-NEXT:     Value (RelocatableAddress): 0x7C
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 18
; SYM-NEXT:       ContainingCsectSymbolIndex: 15
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 0
; SYM-NEXT:       SymbolType: XTY_LD (0x2)
; SYM-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 19
; SYM-NEXT:     Name: storesTIUninit
; SYM-NEXT:     Value (RelocatableAddress): 0x80
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 20
; SYM-NEXT:       SectionLen: 12
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 2
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_DS (0xA)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 21
; SYM-NEXT:     Name: loadsTGInit
; SYM-NEXT:     Value (RelocatableAddress): 0x8C
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 22
; SYM-NEXT:       SectionLen: 12
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 2
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_DS (0xA)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 23
; SYM-NEXT:     Name: TOC
; SYM-NEXT:     Value (RelocatableAddress): 0x98
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 24
; SYM-NEXT:       SectionLen: 0
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 2
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_TC0 (0xF)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 25
; SYM-NEXT:     Name: TIUninit
; SYM-NEXT:     Value (RelocatableAddress): 0x98
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 26
; SYM-NEXT:       SectionLen: 4
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 2
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_TC (0x3)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 27
; SYM-NEXT:     Name: _$TLSML
; SYM-NEXT:     Value (RelocatableAddress): 0x9C
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 28
; SYM-NEXT:       SectionLen: 4
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 2
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_TC (0x3)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 29
; SYM-NEXT:     Name: .TGInit
; SYM-NEXT:     Value (RelocatableAddress): 0xA0
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 30
; SYM-NEXT:       SectionLen: 4
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 2
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_TC (0x3)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 31
; SYM-NEXT:     Name: TGInit
; SYM-NEXT:     Value (RelocatableAddress): 0xA4
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 32
; SYM-NEXT:       SectionLen: 4
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 2
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_TC (0x3)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 33
; SYM-NEXT:     Name: GInit
; SYM-NEXT:     Value (RelocatableAddress): 0xA8
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 34
; SYM-NEXT:       SectionLen: 4
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 2
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_TC (0x3)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 35
; SYM-NEXT:     Name: .tdata
; SYM-NEXT:     Value (RelocatableAddress): 0x0
; SYM-NEXT:     Section: .tdata
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 36
; SYM-NEXT:       SectionLen: 4
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 2
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_TL (0x14)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 37
; SYM-NEXT:     Name: TGInit
; SYM-NEXT:     Value (RelocatableAddress): 0x0
; SYM-NEXT:     Section: .tdata
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 38
; SYM-NEXT:       ContainingCsectSymbolIndex: 35
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 0
; SYM-NEXT:       SymbolType: XTY_LD (0x2)
; SYM-NEXT:       StorageMappingClass: XMC_TL (0x14)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 39
; SYM-NEXT:     Name: TIUninit
; SYM-NEXT:     Value (RelocatableAddress): 0x4
; SYM-NEXT:     Section: .tbss
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 40
; SYM-NEXT:       SectionLen: 4
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 2
; SYM-NEXT:       SymbolType: XTY_CM (0x3)
; SYM-NEXT:       StorageMappingClass: XMC_UL (0x15)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT: ]

; DIS:      {{.*}}aix-tls-xcoff-reloc.ll.tmp.o:	file format aixcoff-rs6000
; DIS:      Disassembly of section .text:
; DIS:      00000000 (idx: 7) .storesTIUninit:
; DIS-NEXT:                                      mflr 0
; DIS-NEXT:                                      stwu 1, -32(1)
; DIS-NEXT:                                      stw 0, 40(1)
; DIS-NEXT:                                      mr 6, 3
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               lwz 7, 0(2)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOC (idx: 25) TIUninit[TC]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               lwz 3, 4(2)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOC (idx: 27) _$TLSML[TC]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               bla 0
; DIS-NEXT: {{0*}}[[#ADDR]]: R_RBA (idx: 1)      .__tls_get_mod[PR]
; DIS-NEXT:                                      add 3, 3, 7
; DIS-NEXT:                                      stw 6, 0(3)
; DIS-NEXT:                                      addi 1, 1, 32
; DIS-NEXT:                                      lwz 0, 8(1)
; DIS-NEXT:                                      mtlr 0
; DIS-NEXT:                                      blr
; DIS:      00000040 (idx: 9) .loadsTGInit:
; DIS-NEXT:                                      mflr 0
; DIS-NEXT:                                      stwu 1, -32(1)
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               lwz 3, 8(2)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOC (idx: 29) .TGInit[TC]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               lwz 4, 12(2)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOC (idx: 31) TGInit[TC]
; DIS-NEXT:                                      stw 0, 40(1)
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               bla 0
; DIS-NEXT: {{0*}}[[#ADDR]]: R_RBA (idx: 3)      .__tls_get_addr[PR]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               lwz 4, 16(2)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOC (idx: 33) GInit[TC]
; DIS-NEXT:                                      lwz 3, 0(3)
; DIS-NEXT:                                      lwz 4, 0(4)
; DIS-NEXT:                                      add 3, 4, 3
; DIS-NEXT:                                      addi 1, 1, 32
; DIS-NEXT:                                      lwz 0, 8(1)
; DIS-NEXT:                                      mtlr 0
; DIS-NEXT:                                      blr
; DIS:      00000078 (idx: 13) const_ivar:
; DIS-NEXT:       78: 00 00 00 06

; DIS:      Disassembly of section .data:
; DIS:      0000007c  (idx: 17) GInit:
; DIS-NEXT:       7c: 00 00 00 01
; DIS:      00000080  (idx: 19) storesTIUninit[DS]:
; DIS-NEXT:       80: 00 00 00 00
; DIS-NEXT: 00000080: R_POS (idx: 7) .storesTIUninit
; DIS-NEXT:       84: 00 00 00 98
; DIS-NEXT: 00000084: R_POS (idx: 23) TOC[TC0]
; DIS-NEXT:       88: 00 00 00 00
; DIS:      0000008c  (idx: 21) loadsTGInit[DS]:
; DIS-NEXT:       8c: 00 00 00 40
; DIS-NEXT: 0000008c: R_POS (idx: 9) .loadsTGInit
; DIS-NEXT:       90: 00 00 00 98
; DIS-NEXT: 00000090: R_POS (idx: 23) TOC[TC0]
; DIS-NEXT:       94: 00 00 00 00
; DIS:      00000098  (idx: 25) TIUninit[TC]:
; DIS-NEXT:       98: 00 00 00 00
; DIS-NEXT: 00000098: R_TLS_LD (idx: 39) TIUninit[UL]
; DIS:      0000009c  (idx: 27) _$TLSML[TC]:
; DIS-NEXT:       9c: 00 00 00 00
; DIS-NEXT: 0000009c: R_TLSML (idx: 27) _$TLSML[TC]
; DIS:      000000a0  (idx: 29) .TGInit[TC]:
; DIS-NEXT:       a0: 00 00 00 00
; DIS-NEXT: 000000a0: R_TLSM (idx: 37) TGInit
; DIS:      000000a4  (idx: 31) TGInit[TC]:
; DIS-NEXT:       a4: 00 00 00 00
; DIS-NEXT: 000000a4: R_TLS (idx: 37) TGInit
; DIS:      000000a8  (idx: 33) GInit[TC]:
; DIS-NEXT:       a8: 00 00 00 7c
; DIS-NEXT: 000000a8: R_POS (idx: 17) GInit

; DIS:      Disassembly of section .tdata:
; DIS:      00000000 (idx: 37) TGInit:
; DIS-NEXT:        0: 00 00 00 01

; DIS:      Disassembly of section .tbss:
; DIS:      00000004 (idx: 39) TIUninit[UL]:
; DIS-NEXT: ...

attributes #0 = { nofree norecurse nounwind willreturn writeonly "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pwr4" "target-features"="-altivec,-bpermd,-crypto,-direct-move,-extdiv,-float128,-htm,-mma,-paired-vector-memops,-power10-vector,-power8-vector,-power9-vector,-spe,-vsx" }
attributes #1 = { norecurse nounwind readonly willreturn "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pwr4" "target-features"="-altivec,-bpermd,-crypto,-direct-move,-extdiv,-float128,-htm,-mma,-paired-vector-memops,-power10-vector,-power8-vector,-power9-vector,-spe,-vsx" }
