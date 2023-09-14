; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false --code-model=large -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --relocs --expand-relocs %t.o | FileCheck --check-prefix=RELOC %s
; RUN: llvm-readobj --syms %t.o | FileCheck --check-prefix=SYM %s
; RUN: llvm-objdump -D -r --symbol-description %t.o | FileCheck --check-prefix=DIS %s

@GInit = global double 1.000000e+00, align 8
; @TIInit is local-dynamic indeed
@TIInit = internal thread_local global i64 1, align 8
@TWInit = weak thread_local global double 1.000000e+00, align 8

; Function Attrs: nofree norecurse nounwind willreturn writeonly
define void @storesTIInit(i64 %Val) #0 {
entry:
  store i64 %Val, ptr @TIInit, align 8
  ret void
}

; Function Attrs: norecurse nounwind readonly willreturn
define double @loadsTWInit() #1 {
entry:
  %0 = load double, ptr @TWInit, align 8
  %1 = load double, ptr @GInit, align 8
  %add = fadd double %0, %1
  ret double %add
}

; RELOC:      File: {{.*}}aix-tls-xcoff-reloc-large.ll.tmp.o
; RELOC-NEXT: Format: aixcoff-rs6000
; RELOC-NEXT: Arch: powerpc
; RELOC-NEXT: AddressSize: 32bit
; RELOC-NEXT: Relocations [
; RELOC-NEXT:   Section (index: 1) .text {
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x16
; RELOC-NEXT:     Symbol: TIInit (19)
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 16
; RELOC-NEXT:     Type: R_TOCU (0x30)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x1A
; RELOC-NEXT:     Symbol: TIInit (19)
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 16
; RELOC-NEXT:     Type: R_TOCL (0x31)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x1E
; RELOC-NEXT:     Symbol: _$TLSML (21)
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 16
; RELOC-NEXT:     Type: R_TOCU (0x30)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x22
; RELOC-NEXT:     Symbol: _$TLSML (21)
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 16
; RELOC-NEXT:     Type: R_TOCL (0x31)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x24
; RELOC-NEXT:     Symbol: .__tls_get_mod (1)
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 26
; RELOC-NEXT:     Type: R_RBA (0x18)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x5E
; RELOC-NEXT:     Symbol: .TWInit (23)
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 16
; RELOC-NEXT:     Type: R_TOCU (0x30)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x62
; RELOC-NEXT:     Symbol: TWInit (25)
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 16
; RELOC-NEXT:     Type: R_TOCU (0x30)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x66
; RELOC-NEXT:     Symbol: .TWInit (23)
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 16
; RELOC-NEXT:     Type: R_TOCL (0x31)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x6A
; RELOC-NEXT:     Symbol: TWInit (25)
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 16
; RELOC-NEXT:     Type: R_TOCL (0x31)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x6C
; RELOC-NEXT:     Symbol: .__tls_get_addr (3)
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 26
; RELOC-NEXT:     Type: R_RBA (0x18)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x76
; RELOC-NEXT:     Symbol: GInit (27)
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 16
; RELOC-NEXT:     Type: R_TOCU (0x30)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x7A
; RELOC-NEXT:     Symbol: GInit (27)
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 16
; RELOC-NEXT:     Type: R_TOCL (0x31)
; RELOC-NEXT:   }
; RELOC-NEXT: }
; RELOC-NEXT: Section (index: 2) .data {
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0xA0
; RELOC-NEXT:   Symbol: .storesTIInit (7)
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_POS (0x0)
; RELOC-NEXT: }
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0xA4
; RELOC-NEXT:   Symbol: TOC (17)
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_POS (0x0)
; RELOC-NEXT: }
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0xAC
; RELOC-NEXT:   Symbol: .loadsTWInit (9)
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_POS (0x0)
; RELOC-NEXT: }
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0xB0
; RELOC-NEXT:   Symbol: TOC (17)
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_POS (0x0)
; RELOC-NEXT: }
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0xB8
; RELOC-NEXT:   Symbol: TIInit (29)
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_TLS_LD (0x22)
; RELOC-NEXT: }
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0xBC
; RELOC-NEXT:   Symbol: _$TLSML (21)
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_TLSML (0x25)
; RELOC-NEXT: }
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0xC0
; RELOC-NEXT:   Symbol: TWInit (31)
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_TLSM (0x24)
; RELOC-NEXT: }
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0xC4
; RELOC-NEXT:   Symbol: TWInit (31)
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_TLS (0x20)
; RELOC-NEXT: }
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0xC8
; RELOC-NEXT:   Symbol: GInit (11)
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_POS (0x0)
; RELOC-NEXT: }
; RELOC-NEXT: }
; RELOC-NEXT: ]

; SYM:      File: {{.*}}aix-tls-xcoff-reloc-large.ll.tmp.o
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
; SYM-NEXT:       SectionLen: 148
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
; SYM-NEXT:     Name: .storesTIInit
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
; SYM-NEXT:     Name: .loadsTWInit
; SYM-NEXT:     Value (RelocatableAddress): 0x50
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
; SYM-NEXT:     Name: GInit
; SYM-NEXT:     Value (RelocatableAddress): 0x98
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 12
; SYM-NEXT:       SectionLen: 8
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 3
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 13
; SYM-NEXT:     Name: storesTIInit
; SYM-NEXT:     Value (RelocatableAddress): 0xA0
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 14
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
; SYM-NEXT:     Index: 15
; SYM-NEXT:     Name: loadsTWInit
; SYM-NEXT:     Value (RelocatableAddress): 0xAC
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 16
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
; SYM-NEXT:     Index: 17
; SYM-NEXT:     Name: TOC
; SYM-NEXT:     Value (RelocatableAddress): 0xB8
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 18
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
; SYM-NEXT:     Index: 19
; SYM-NEXT:     Name: TIInit
; SYM-NEXT:     Value (RelocatableAddress): 0xB8
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 20
; SYM-NEXT:       SectionLen: 4
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 2
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_TE (0x16)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 21
; SYM-NEXT:     Name: _$TLSML
; SYM-NEXT:     Value (RelocatableAddress): 0xBC
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 22
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
; SYM-NEXT:     Index: 23
; SYM-NEXT:     Name: .TWInit
; SYM-NEXT:     Value (RelocatableAddress): 0xC0
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 24
; SYM-NEXT:       SectionLen: 4
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 2
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_TE (0x16)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 25
; SYM-NEXT:     Name: TWInit
; SYM-NEXT:     Value (RelocatableAddress): 0xC4
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
; SYM-NEXT:       StorageMappingClass: XMC_TE (0x16)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 27
; SYM-NEXT:     Name: GInit
; SYM-NEXT:     Value (RelocatableAddress): 0xC8
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
; SYM-NEXT:       StorageMappingClass: XMC_TE (0x16)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 29
; SYM-NEXT:     Name: TIInit
; SYM-NEXT:     Value (RelocatableAddress): 0x0
; SYM-NEXT:     Section: .tdata
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 30
; SYM-NEXT:       SectionLen: 8
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 3
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_TL (0x14)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 31
; SYM-NEXT:     Name: TWInit
; SYM-NEXT:     Value (RelocatableAddress): 0x8
; SYM-NEXT:     Section: .tdata
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_WEAKEXT (0x6F)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 32
; SYM-NEXT:       SectionLen: 8
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 3
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_TL (0x14)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT: ]

; DIS:      {{.*}}aix-tls-xcoff-reloc-large.ll.tmp.o:	file format aixcoff-rs6000
; DIS:      Disassembly of section .text:
; DIS:      00000000 (idx: 7) .storesTIInit:
; DIS-NEXT:                                       mflr 0
; DIS-NEXT:                                       stwu 1, -32(1)
; DIS-NEXT:                                       stw 0, 40(1)
; DIS-NEXT:                                       mr 7, 3
; DIS-NEXT:                                       mr 6, 4
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                addis 3, 2, 0
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOCU (idx: 19) TIInit[TE]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                lwz 8, 0(3)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOCL (idx: 19) TIInit[TE]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                addis 3, 2, 0
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOCU (idx: 21) _$TLSML[TC]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                lwz 3, 4(3)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOCL (idx: 21) _$TLSML[TC]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                bla 0
; DIS-NEXT: {{0*}}[[#ADDR]]: R_RBA  (idx: 1)      .__tls_get_mod[PR]
; DIS-NEXT:                                       add 3, 3, 8
; DIS-NEXT:                                       stw 6, 4(3)
; DIS-NEXT:                                       stw 7, 0(3)
; DIS-NEXT:                                       addi 1, 1, 32
; DIS-NEXT:                                       lwz 0, 8(1)
; DIS-NEXT:                                       mtlr 0
; DIS-NEXT:                                       blr
; DIS:      00000050 (idx: 9) .loadsTWInit:
; DIS-NEXT:                                       mflr 0
; DIS-NEXT:                                       stwu 1, -32(1)
; DIS-NEXT:                                       stw 0, 40(1)
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                addis 3, 2, 0
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOCU (idx: 23) .TWInit[TE]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                addis 4, 2, 0
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOCU (idx: 25) TWInit[TE]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                lwz 3, 8(3)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOCL (idx: 23) .TWInit[TE]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                lwz 4, 12(4)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOCL (idx: 25) TWInit[TE]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                bla 0
; DIS-NEXT: {{0*}}[[#ADDR]]: R_RBA  (idx: 3)      .__tls_get_addr[PR]
; DIS-NEXT:                                       lfd 0, 0(3)
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                addis 3, 2, 0
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOCU (idx: 27) GInit[TE]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                lwz 3, 16(3)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOCL (idx: 27) GInit[TE]
; DIS-NEXT:                                       lfd 1, 0(3)
; DIS-NEXT:                                       fadd 1, 0, 1
; DIS-NEXT:                                       addi 1, 1, 32
; DIS-NEXT:                                       lwz 0, 8(1)
; DIS-NEXT:                                       mtlr 0
; DIS-NEXT:                                       blr

; DIS:      Disassembly of section .data:
; DIS:      00000098  (idx: 11) GInit[RW]:
; DIS-NEXT:       98: 3f f0 00 00
; DIS-NEXT:       9c: 00 00 00 00
; DIS:      000000a0  (idx: 13) storesTIInit[DS]:
; DIS-NEXT:       a0: 00 00 00 00
; DIS-NEXT: 000000a0: R_POS (idx: 7) .storesTIInit
; DIS-NEXT:       a4: 00 00 00 b8
; DIS-NEXT: 000000a4: R_POS (idx: 17) TOC[TC0]
; DIS-NEXT:       a8: 00 00 00 00
; DIS:      000000ac  (idx: 15) loadsTWInit[DS]:
; DIS-NEXT:       ac: 00 00 00 50
; DIS-NEXT: 000000ac: R_POS (idx: 9) .loadsTWInit
; DIS-NEXT:       b0: 00 00 00 b8
; DIS-NEXT: 000000b0: R_POS (idx: 17) TOC[TC0]
; DIS-NEXT:       b4: 00 00 00 00
; DIS:      000000b8  (idx: 19) TIInit[TE]:
; DIS-NEXT:       b8: 00 00 00 00
; DIS-NEXT: 000000b8: R_TLS_LD (idx: 29) TIInit[TL]
; DIS:      000000bc  (idx: 21) _$TLSML[TC]:
; DIS-NEXT:       bc: 00 00 00 00
; DIS-NEXT: 000000bc: R_TLSML (idx: 21) _$TLSML[TC]
; DIS:      000000c0  (idx: 23) .TWInit[TE]:
; DIS-NEXT:       c0: 00 00 00 00
; DIS-NEXT: 000000c0: R_TLSM (idx: 31) TWInit[TL]
; DIS:      000000c4  (idx: 25) TWInit[TE]:
; DIS-NEXT:       c4: 00 00 00 08
; DIS-NEXT: 000000c4: R_TLS (idx: 31) TWInit[TL]
; DIS:      000000c8  (idx: 27) GInit[TE]:
; DIS-NEXT:       c8: 00 00 00 98
; DIS-NEXT: 000000c8: R_POS (idx: 11) GInit[RW]

; DIS:      Disassembly of section .tdata:
; DIS:      00000000  (idx: 29) TIInit[TL]:
; DIS-NEXT:        0: 00 00 00 00
; DIS-NEXT:        4: 00 00 00 01
; DIS:      00000008  (idx: 31) TWInit[TL]:
; DIS-NEXT:        8: 3f f0 00 00
; DIS-NEXT:        c: 00 00 00 00

attributes #0 = { nofree norecurse nounwind willreturn writeonly "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pwr4" "target-features"="-altivec,-bpermd,-crypto,-direct-move,-extdiv,-float128,-htm,-mma,-paired-vector-memops,-power10-vector,-power8-vector,-power9-vector,-spe,-vsx" }
attributes #1 = { norecurse nounwind readonly willreturn "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pwr4" "target-features"="-altivec,-bpermd,-crypto,-direct-move,-extdiv,-float128,-htm,-mma,-paired-vector-memops,-power10-vector,-power8-vector,-power9-vector,-spe,-vsx" }
