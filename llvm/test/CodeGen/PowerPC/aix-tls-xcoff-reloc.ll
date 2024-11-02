; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false -data-sections=false -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --relocs --expand-relocs %t.o | FileCheck -D#NFA=2 --check-prefix=RELOC %s
; RUN: llvm-readobj --syms %t.o | FileCheck -D#NFA=2 --check-prefix=SYM %s
; RUN: llvm-objdump -D -r --symbol-description %t.o | FileCheck -D#NFA=2 --check-prefix=DIS %s

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

; RELOC:      File:
; RELOC-NEXT: Format: aixcoff-rs6000
; RELOC-NEXT: Arch: powerpc
; RELOC-NEXT: AddressSize: 32bit
; RELOC-NEXT: Relocations [
; RELOC-NEXT:   Section (index: 1) .text {
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0xE
; RELOC-NEXT:     Symbol: _$TLSML ([[#NFA+25]])
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 16
; RELOC-NEXT:     Type: R_TOC (0x3)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x14
; RELOC-NEXT:     Symbol: .__tls_get_mod ([[#NFA+1]])
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 26
; RELOC-NEXT:     Type: R_RBA (0x18)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x1A
; RELOC-NEXT:     Symbol: TIUninit ([[#NFA+27]])
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 16
; RELOC-NEXT:     Type: R_TOC (0x3)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x3A
; RELOC-NEXT:     Symbol: .TGInit ([[#NFA+29]])
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 16
; RELOC-NEXT:     Type: R_TOC (0x3)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x3E
; RELOC-NEXT:     Symbol: TGInit ([[#NFA+31]])
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 16
; RELOC-NEXT:     Type: R_TOC (0x3)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x44
; RELOC-NEXT:     Symbol: .__tls_get_addr ([[#NFA+3]])
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 26
; RELOC-NEXT:     Type: R_RBA (0x18)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x4A
; RELOC-NEXT:     Symbol: GInit ([[#NFA+33]])
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 16
; RELOC-NEXT:     Type: R_TOC (0x3)
; RELOC-NEXT:   }
; RELOC-NEXT: }
; RELOC-NEXT: Section (index: 2) .data {
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0x70
; RELOC-NEXT:   Symbol: .storesTIUninit ([[#NFA+7]])
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_POS (0x0)
; RELOC-NEXT: }
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0x74
; RELOC-NEXT:   Symbol: TOC ([[#NFA+23]])
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_POS (0x0)
; RELOC-NEXT: }
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0x7C
; RELOC-NEXT:   Symbol: .loadsTGInit ([[#NFA+9]])
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_POS (0x0)
; RELOC-NEXT: }
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0x80
; RELOC-NEXT:   Symbol: TOC ([[#NFA+23]])
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_POS (0x0)
; RELOC-NEXT: }
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0x88
; RELOC-NEXT:   Symbol: _$TLSML ([[#NFA+25]])
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_TLSML (0x25)
; RELOC-NEXT: }
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0x8C
; RELOC-NEXT:   Symbol: TIUninit ([[#NFA+39]])
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_TLS_LD (0x22)
; RELOC-NEXT: }
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0x90
; RELOC-NEXT:   Symbol: TGInit ([[#NFA+37]])
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_TLSM (0x24)
; RELOC-NEXT: }
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0x94
; RELOC-NEXT:   Symbol: TGInit ([[#NFA+37]])
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_TLS (0x20)
; RELOC-NEXT: }
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0x98
; RELOC-NEXT:   Symbol: GInit ([[#NFA+17]])
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_POS (0x0)
; RELOC-NEXT: }
; RELOC-NEXT: }
; RELOC-NEXT: ]

; SYM:      File:
; SYM-NEXT: Format: aixcoff-rs6000
; SYM-NEXT: Arch: powerpc
; SYM-NEXT: AddressSize: 32bit
; SYM-NEXT: Symbols [
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 0
; SYM-NEXT:     Name: .file
; SYM-NEXT:     Value (SymbolTableIndex): 0x0
; SYM-NEXT:     Section: N_DEBUG
; SYM-NEXT:     Source Language ID: TB_CPLUSPLUS (0x9)
; SYM-NEXT:     CPU Version ID: TCPU_COM (0x3)
; SYM-NEXT:     StorageClass: C_FILE (0x67)
; SYM-NEXT:     NumberOfAuxEntries: 2
; SYM:        Symbol {
; SYM-NEXT:     Index: [[#NFA+1]]
; SYM-NEXT:     Name: .__tls_get_mod
; SYM-NEXT:     Value (RelocatableAddress): 0x0
; SYM-NEXT:     Section: N_UNDEF
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#NFA+2]]
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
; SYM-NEXT:     Index: [[#NFA+3]]
; SYM-NEXT:     Name: .__tls_get_addr
; SYM-NEXT:     Value (RelocatableAddress): 0x0
; SYM-NEXT:     Section: N_UNDEF
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 6
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
; SYM-NEXT:     Index: [[#NFA+5]]
; SYM-NEXT:     Name:
; SYM-NEXT:     Value (RelocatableAddress): 0x0
; SYM-NEXT:     Section: .text
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#NFA+6]]
; SYM-NEXT:       SectionLen: 104
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
; SYM-NEXT:     Index: [[#NFA+7]]
; SYM-NEXT:     Name: .storesTIUninit
; SYM-NEXT:     Value (RelocatableAddress): 0x0
; SYM-NEXT:     Section: .text
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#NFA+8]]
; SYM-NEXT:       ContainingCsectSymbolIndex: [[#NFA+5]]
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
; SYM-NEXT:     Index: [[#NFA+9]]
; SYM-NEXT:     Name: .loadsTGInit
; SYM-NEXT:     Value (RelocatableAddress): 0x30
; SYM-NEXT:     Section: .text
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#NFA+10]]
; SYM-NEXT:       ContainingCsectSymbolIndex: [[#NFA+5]]
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
; SYM-NEXT:     Index: [[#NFA+11]]
; SYM-NEXT:     Name: .rodata
; SYM-NEXT:     Value (RelocatableAddress): 0x68
; SYM-NEXT:     Section: .text
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#NFA+12]]
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
; SYM-NEXT:     Index: [[#NFA+13]]
; SYM-NEXT:     Name: const_ivar
; SYM-NEXT:     Value (RelocatableAddress): 0x68
; SYM-NEXT:     Section: .text
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#NFA+14]]
; SYM-NEXT:       ContainingCsectSymbolIndex: [[#NFA+11]]
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
; SYM-NEXT:     Index: [[#NFA+15]]
; SYM-NEXT:     Name: .data
; SYM-NEXT:     Value (RelocatableAddress): 0x6C
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#NFA+16]]
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
; SYM-NEXT:     Index: [[#NFA+17]]
; SYM-NEXT:     Name: GInit
; SYM-NEXT:     Value (RelocatableAddress): 0x6C
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#NFA+18]]
; SYM-NEXT:       ContainingCsectSymbolIndex: [[#NFA+15]]
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
; SYM-NEXT:     Index: [[#NFA+19]]
; SYM-NEXT:     Name: storesTIUninit
; SYM-NEXT:     Value (RelocatableAddress): 0x70
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#NFA+20]]
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
; SYM-NEXT:     Index: [[#NFA+21]]
; SYM-NEXT:     Name: loadsTGInit
; SYM-NEXT:     Value (RelocatableAddress): 0x7C
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#NFA+22]]
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
; SYM-NEXT:     Index: [[#NFA+23]]
; SYM-NEXT:     Name: TOC
; SYM-NEXT:     Value (RelocatableAddress): 0x88
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#NFA+24]]
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
; SYM-NEXT:     Index: [[#NFA+25]]
; SYM-NEXT:     Name: _$TLSML
; SYM-NEXT:     Value (RelocatableAddress): 0x88
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#NFA+26]]
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
; SYM-NEXT:     Index: [[#NFA+27]]
; SYM-NEXT:     Name: TIUninit
; SYM-NEXT:     Value (RelocatableAddress): 0x8C
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#NFA+28]]
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
; SYM-NEXT:     Index: [[#NFA+29]]
; SYM-NEXT:     Name: .TGInit
; SYM-NEXT:     Value (RelocatableAddress): 0x90
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#NFA+30]]
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
; SYM-NEXT:     Index: [[#NFA+31]]
; SYM-NEXT:     Name: TGInit
; SYM-NEXT:     Value (RelocatableAddress): 0x94
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#NFA+32]]
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
; SYM-NEXT:     Index: [[#NFA+33]]
; SYM-NEXT:     Name: GInit
; SYM-NEXT:     Value (RelocatableAddress): 0x98
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#NFA+34]]
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
; SYM-NEXT:     Index: [[#NFA+35]]
; SYM-NEXT:     Name: .tdata
; SYM-NEXT:     Value (RelocatableAddress): 0x0
; SYM-NEXT:     Section: .tdata
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#NFA+36]]
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
; SYM-NEXT:     Index: [[#NFA+37]]
; SYM-NEXT:     Name: TGInit
; SYM-NEXT:     Value (RelocatableAddress): 0x0
; SYM-NEXT:     Section: .tdata
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#NFA+38]]
; SYM-NEXT:       ContainingCsectSymbolIndex: [[#NFA+35]]
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
; SYM-NEXT:     Index: [[#NFA+39]]
; SYM-NEXT:     Name: TIUninit
; SYM-NEXT:     Value (RelocatableAddress): 0x4
; SYM-NEXT:     Section: .tbss
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#NFA+40]]
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
; DIS:      00000000 (idx: [[#NFA+7]]) .storesTIUninit:
; DIS-NEXT:                                      mflr 0
; DIS-NEXT:                                      stwu 1, -32(1)
; DIS-NEXT:                                      mr 6, 3
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               lwz 3, 0(2)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOC (idx: [[#NFA+25]]) _$TLSML[TC]
; DIS-NEXT:                                      stw 0, 40(1)
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               bla 0x0
; DIS-NEXT: {{0*}}[[#ADDR]]: R_RBA (idx: [[#NFA+1]]) .__tls_get_mod[PR]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               lwz 4, 4(2)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOC (idx: [[#NFA+27]]) TIUninit[TC]
; DIS-NEXT:                                      stwx 6, 3, 4
; DIS-NEXT:                                      addi 1, 1, 32
; DIS-NEXT:                                      lwz 0, 8(1)
; DIS-NEXT:                                      mtlr 0
; DIS-NEXT:                                      blr
; DIS:      00000030 (idx: [[#NFA+9]]) .loadsTGInit:
; DIS-NEXT:                                      mflr 0
; DIS-NEXT:                                      stwu 1, -32(1)
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               lwz 3, 8(2)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOC (idx: [[#NFA+29]]) .TGInit[TC]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               lwz 4, 12(2)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOC (idx: [[#NFA+31]]) TGInit[TC]
; DIS-NEXT:                                      stw 0, 40(1)
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               bla 0
; DIS-NEXT: {{0*}}[[#ADDR]]: R_RBA (idx: [[#NFA+3]])      .__tls_get_addr[PR]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               lwz 4, 16(2)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOC (idx: [[#NFA+33]]) GInit[TC]
; DIS-NEXT:                                      lwz 3, 0(3)
; DIS-NEXT:                                      lwz 4, 0(4)
; DIS-NEXT:                                      add 3, 4, 3
; DIS-NEXT:                                      addi 1, 1, 32
; DIS-NEXT:                                      lwz 0, 8(1)
; DIS-NEXT:                                      mtlr 0
; DIS-NEXT:                                      blr
; DIS:      00000068 (idx: [[#NFA+13]]) const_ivar:
; DIS-NEXT:       68: 00 00 00 06

; DIS:      Disassembly of section .data:
; DIS:      0000006c  (idx: [[#NFA+17]]) GInit:
; DIS-NEXT:       6c: 00 00 00 01
; DIS:      00000070  (idx: [[#NFA+19]]) storesTIUninit[DS]:
; DIS-NEXT:       70: 00 00 00 00
; DIS-NEXT: 00000070: R_POS (idx: [[#NFA+7]]) .storesTIUninit
; DIS-NEXT:       74: 00 00 00 88
; DIS-NEXT: 00000074: R_POS (idx: [[#NFA+23]]) TOC[TC0]
; DIS-NEXT:       78: 00 00 00 00
; DIS:      0000007c  (idx: [[#NFA+21]]) loadsTGInit[DS]:
; DIS-NEXT:       7c: 00 00 00 30
; DIS-NEXT: 0000007c: R_POS (idx: [[#NFA+9]]) .loadsTGInit
; DIS-NEXT:       80: 00 00 00 88
; DIS-NEXT: 00000080: R_POS (idx: [[#NFA+23]]) TOC[TC0]
; DIS-NEXT:       84: 00 00 00 00
; DIS:      00000088 (idx: [[#NFA+25]]) _$TLSML[TC]:
; DIS-NEXT:       88: 00 00 00 00
; DIS-NEXT: 00000088: R_TLSML (idx: [[#NFA+25]]) _$TLSML[TC]
; DIS:      0000008c  (idx: [[#NFA+27]]) TIUninit[TC]:
; DIS-NEXT:       8c: 00 00 00 04
; DIS-NEXT: 0000008c: R_TLS_LD (idx: [[#NFA+39]]) TIUninit[UL]
; DIS:      00000090  (idx: [[#NFA+29]]) .TGInit[TC]:
; DIS-NEXT:       90: 00 00 00 00
; DIS-NEXT: 00000090: R_TLSM (idx: [[#NFA+37]]) TGInit
; DIS:      00000094  (idx: [[#NFA+31]]) TGInit[TC]:
; DIS-NEXT:       94: 00 00 00 00
; DIS-NEXT: 00000094: R_TLS (idx: [[#NFA+37]]) TGInit
; DIS:      00000098  (idx: [[#NFA+33]]) GInit[TC]:
; DIS-NEXT:       98: 00 00 00 6c
; DIS-NEXT: 00000098: R_POS (idx: [[#NFA+17]]) GInit

; DIS:      Disassembly of section .tdata:
; DIS:      00000000 (idx: [[#NFA+37]]) TGInit:
; DIS-NEXT:        0: 00 00 00 01

; DIS:      Disassembly of section .tbss:
; DIS:      00000004 (idx: [[#NFA+39]]) TIUninit[UL]:
; DIS-NEXT: ...

attributes #0 = { nofree norecurse nounwind willreturn writeonly "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pwr4" "target-features"="-altivec,-bpermd,-crypto,-direct-move,-extdiv,-float128,-htm,-mma,-paired-vector-memops,-power10-vector,-power8-vector,-power9-vector,-spe,-vsx" }
attributes #1 = { norecurse nounwind readonly willreturn "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pwr4" "target-features"="-altivec,-bpermd,-crypto,-direct-move,-extdiv,-float128,-htm,-mma,-paired-vector-memops,-power10-vector,-power8-vector,-power9-vector,-spe,-vsx" }
