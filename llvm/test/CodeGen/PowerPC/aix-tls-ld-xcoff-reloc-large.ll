; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false --code-model=large -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --relocs --expand-relocs %t.o | FileCheck -D#NFA=2 --check-prefix=RELOC %s
; RUN: llvm-readobj --syms %t.o | FileCheck -D#NFA=2 --check-prefix=SYM %s
; RUN: llvm-objdump -D -r --symbol-description %t.o | FileCheck -D#NFA=2 --check-prefix=DIS %s

@ThreadLocalVarInit = thread_local(localdynamic) global i64 1, align 8
@IThreadLocalVarUninit = internal thread_local(localdynamic) global i64 0, align 8
@IThreadLocalVarUninit2 = internal thread_local(localdynamic) global i64 0, align 8
declare nonnull ptr @llvm.threadlocal.address.p0(ptr nonnull)

define void @storeITLUninit(i64 noundef %x) {
entry:
  %0 = tail call align 8 ptr @llvm.threadlocal.address.p0(ptr align 8 @IThreadLocalVarUninit)
  store i64 %x, ptr %0, align 8
  ret void
}

define i64 @loadTLInit() {
entry:
  %0 = tail call align 8 ptr @llvm.threadlocal.address.p0(ptr align 8 @ThreadLocalVarInit)
  %1 = load i64, ptr %0, align 8
  ret i64 %1
}

define signext i64 @loadTLUninit() {
entry:
  %0 = tail call align 8 ptr @llvm.threadlocal.address.p0(ptr align 8 @IThreadLocalVarUninit)
  store i64 1, ptr %0, align 8
  %1 = tail call align 8 ptr @llvm.threadlocal.address.p0(ptr align 8 @IThreadLocalVarUninit2)
  %2 = load i64, ptr %1, align 8
  %add = add nsw i64 %2, 1
  ret i64 %add
}

; RELOC:      File: {{.*}}aix-tls-ld-xcoff-reloc-large.ll.tmp.o
; RELOC-NEXT: Format: aix5coff64-rs6000
; RELOC-NEXT: Arch: powerpc64
; RELOC-NEXT: AddressSize: 64bit
; RELOC-NEXT: Relocations [
; RELOC:      Virtual Address: 0xE
; RELOC-NEXT:      Symbol: _$TLSML ([[#NFA+19]])
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x12
; RELOC-NEXT:      Symbol: IThreadLocalVarUninit ([[#NFA+21]])
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x1A
; RELOC-NEXT:      Symbol: _$TLSML ([[#NFA+19]])
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x1C
; RELOC-NEXT:      Symbol: .__tls_get_mod ([[#NFA+1]])
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 26
; RELOC-NEXT:      Type: R_RBA (0x18)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x22
; RELOC-NEXT:      Symbol: IThreadLocalVarUninit ([[#NFA+21]])
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x4A
; RELOC-NEXT:      Symbol: _$TLSML ([[#NFA+19]])
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x4E
; RELOC-NEXT:      Symbol: ThreadLocalVarInit ([[#NFA+23]])
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x56
; RELOC-NEXT:      Symbol: _$TLSML ([[#NFA+19]])
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x58
; RELOC-NEXT:      Symbol: .__tls_get_mod ([[#NFA+1]])
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 26
; RELOC-NEXT:      Type: R_RBA (0x18)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x5E
; RELOC-NEXT:      Symbol: ThreadLocalVarInit ([[#NFA+23]])
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x8A
; RELOC-NEXT:      Symbol: _$TLSML ([[#NFA+19]])
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x8E
; RELOC-NEXT:      Symbol: IThreadLocalVarUninit ([[#NFA+21]])
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x96
; RELOC-NEXT:      Symbol: _$TLSML ([[#NFA+19]])
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x98
; RELOC-NEXT:      Symbol: .__tls_get_mod (3)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 26
; RELOC-NEXT:      Type: R_RBA (0x18)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x9E
; RELOC-NEXT:      Symbol: IThreadLocalVarUninit ([[#NFA+21]])
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0xAA
; RELOC-NEXT:      Symbol: IThreadLocalVarUninit2 ([[#NFA+25]])
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0xAE
; RELOC-NEXT:      Symbol: IThreadLocalVarUninit2 ([[#NFA+25]])
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x110
; RELOC-NEXT:      Symbol: _$TLSML ([[#NFA+19]])
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_TLSML (0x25)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x118
; RELOC-NEXT:      Symbol: IThreadLocalVarUninit ([[#NFA+29]])
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_TLS_LD (0x22)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x120
; RELOC-NEXT:      Symbol: ThreadLocalVarInit ([[#NFA+27]])
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_TLS_LD (0x22)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x128
; RELOC-NEXT:      Symbol: IThreadLocalVarUninit2 ([[#NFA+31]])
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_TLS_LD (0x22)
; RELOC-NEXT:    }

; SYM:      File: {{.*}}aix-tls-ld-xcoff-reloc-large.ll.tmp.o
; SYM-NEXT: Format: aix5coff64-rs6000
; SYM-NEXT: Arch: powerpc64
; SYM-NEXT: AddressSize: 64bit
; SYM-NEXT: Symbols [
; SYM:    Index: [[#NFA+19]]
; SYM-NEXT:    Name: _$TLSML
; SYM-NEXT:    Value (RelocatableAddress): 0x110
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:    NumberOfAuxEntries: 1
; SYM-NEXT:    CSECT Auxiliary Entry {
; SYM-NEXT:      Index: [[#NFA+20]]
; SYM-NEXT:      SectionLen: 8
; SYM-NEXT:      ParameterHashIndex: 0x0
; SYM-NEXT:      TypeChkSectNum: 0x0
; SYM-NEXT:      SymbolAlignmentLog2: 3
; SYM-NEXT:      SymbolType: XTY_SD (0x1)
; SYM-NEXT:      StorageMappingClass: XMC_TC (0x3)
; SYM-NEXT:      Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:    }
; SYM-NEXT:  }
; SYM:    Index: [[#NFA+21]]
; SYM-NEXT:    Name: IThreadLocalVarUninit
; SYM-NEXT:    Value (RelocatableAddress): 0x118
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:    NumberOfAuxEntries: 1
; SYM-NEXT:    CSECT Auxiliary Entry {
; SYM-NEXT:      Index: [[#NFA+22]]
; SYM-NEXT:      SectionLen: 8
; SYM-NEXT:      ParameterHashIndex: 0x0
; SYM-NEXT:      TypeChkSectNum: 0x0
; SYM-NEXT:      SymbolAlignmentLog2: 3
; SYM-NEXT:      SymbolType: XTY_SD (0x1)
; SYM-NEXT:      StorageMappingClass: XMC_TE (0x16)
; SYM-NEXT:      Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:    }
; SYM-NEXT:  }
; SYM:    Index: [[#NFA+23]]
; SYM-NEXT:    Name: ThreadLocalVarInit
; SYM-NEXT:    Value (RelocatableAddress): 0x120
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:    NumberOfAuxEntries: 1
; SYM-NEXT:    CSECT Auxiliary Entry {
; SYM-NEXT:      Index: [[#NFA+24]]
; SYM-NEXT:      SectionLen: 8
; SYM-NEXT:      ParameterHashIndex: 0x0
; SYM-NEXT:      TypeChkSectNum: 0x0
; SYM-NEXT:      SymbolAlignmentLog2: 3
; SYM-NEXT:      SymbolType: XTY_SD (0x1)
; SYM-NEXT:      StorageMappingClass: XMC_TE (0x16)
; SYM-NEXT:      Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:    }
; SYM-NEXT:  }
; SYM:    Index: [[#NFA+25]]
; SYM-NEXT:    Name: IThreadLocalVarUninit2
; SYM-NEXT:    Value (RelocatableAddress): 0x128
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:    NumberOfAuxEntries: 1
; SYM-NEXT:    CSECT Auxiliary Entry {
; SYM-NEXT:      Index: [[#NFA+26]]
; SYM-NEXT:      SectionLen: 8
; SYM-NEXT:      ParameterHashIndex: 0x0
; SYM-NEXT:      TypeChkSectNum: 0x0
; SYM-NEXT:      SymbolAlignmentLog2: 3
; SYM-NEXT:      SymbolType: XTY_SD (0x1)
; SYM-NEXT:      StorageMappingClass: XMC_TE (0x16)
; SYM-NEXT:      Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:    }
; SYM-NEXT:  }
; SYM:    Index: [[#NFA+27]]
; SYM-NEXT:    Name: ThreadLocalVarInit
; SYM-NEXT:    Value (RelocatableAddress): 0x0
; SYM-NEXT:    Section: .tdata
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_EXT (0x2)
; SYM-NEXT:    NumberOfAuxEntries: 1
; SYM-NEXT:    CSECT Auxiliary Entry {
; SYM-NEXT:      Index: [[#NFA+28]]
; SYM-NEXT:      SectionLen: 8
; SYM-NEXT:      ParameterHashIndex: 0x0
; SYM-NEXT:      TypeChkSectNum: 0x0
; SYM-NEXT:      SymbolAlignmentLog2: 3
; SYM-NEXT:      SymbolType: XTY_SD (0x1)
; SYM-NEXT:      StorageMappingClass: XMC_TL (0x14)
; SYM-NEXT:      Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:    }
; SYM-NEXT:  }
; SYM:    Index: [[#NFA+29]]
; SYM-NEXT:    Name: IThreadLocalVarUninit
; SYM-NEXT:    Value (RelocatableAddress): 0x8
; SYM-NEXT:    Section: .tbss
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:    NumberOfAuxEntries: 1
; SYM-NEXT:    CSECT Auxiliary Entry {
; SYM-NEXT:      Index: [[#NFA+30]]
; SYM-NEXT:      SectionLen: 8
; SYM-NEXT:      ParameterHashIndex: 0x0
; SYM-NEXT:      TypeChkSectNum: 0x0
; SYM-NEXT:      SymbolAlignmentLog2: 3
; SYM-NEXT:      SymbolType: XTY_CM (0x3)
; SYM-NEXT:      StorageMappingClass: XMC_UL (0x15)
; SYM-NEXT:      Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:    }
; SYM-NEXT:  }
; SYM:    Index: [[#NFA+31]]
; SYM-NEXT:    Name: IThreadLocalVarUninit2
; SYM-NEXT:    Value (RelocatableAddress): 0x10
; SYM-NEXT:    Section: .tbss
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:    NumberOfAuxEntries: 1
; SYM-NEXT:    CSECT Auxiliary Entry {
; SYM-NEXT:      Index: [[#NFA+32]]
; SYM-NEXT:      SectionLen: 8
; SYM-NEXT:      ParameterHashIndex: 0x0
; SYM-NEXT:      TypeChkSectNum: 0x0
; SYM-NEXT:      SymbolAlignmentLog2: 3
; SYM-NEXT:      SymbolType: XTY_CM (0x3)
; SYM-NEXT:      StorageMappingClass: XMC_UL (0x15)
; SYM-NEXT:      Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:    }
; SYM-NEXT:  }

; DIS:      {{.*}}aix-tls-ld-xcoff-reloc-large.ll.tmp.o:	file format aix5coff64-rs6000
; DIS:      Disassembly of section .data:
; DIS:      0000000000000110 (idx: [[#NFA+19]]) _$TLSML[TC]:
; DIS-NEXT:     110: 00 00 00 00
; DIS-NEXT:     0000000000000110:  R_TLSML	(idx: [[#NFA+19]]) _$TLSML[TC]
; DIS-NEXT:     114: 00 00 00 00
; DIS:      0000000000000118 (idx: [[#NFA+21]]) IThreadLocalVarUninit[TE]:
; DIS-NEXT:     118: 00 00 00 00
; DIS-NEXT:     0000000000000118:  R_TLS_LD	(idx: [[#NFA+29]]) IThreadLocalVarUninit[UL]
; DIS-NEXT:     11c: 00 00 00 08
; DIS:      0000000000000120 (idx: [[#NFA+23]]) ThreadLocalVarInit[TE]:
; DIS-NEXT:     120: 00 00 00 00
; DIS-NEXT:     0000000000000120:  R_TLS_LD	(idx: [[#NFA+27]]) ThreadLocalVarInit[TL]
; DIS-NEXT:     124: 00 00 00 00
; DIS:      0000000000000128 (idx: [[#NFA+25]]) IThreadLocalVarUninit2[TE]:
; DIS-NEXT:     128: 00 00 00 00
; DIS-NEXT:     0000000000000128:  R_TLS_LD	(idx: [[#NFA+31]]) IThreadLocalVarUninit2[UL]
; DIS-NEXT:     12c: 00 00 00 10

; DIS:      Disassembly of section .tdata:
; DIS:      0000000000000000 (idx: [[#NFA+27]]) ThreadLocalVarInit[TL]:
; DIS-NEXT:        0: 00 00 00 00
; DIS-NEXT:        4: 00 00 00 01

; DIS:      Disassembly of section .tbss:
; DIS:      0000000000000008 (idx: [[#NFA+29]]) IThreadLocalVarUninit[UL]:
; DIS-NEXT: ...
; DIS:      0000000000000010 (idx: [[#NFA+31]]) IThreadLocalVarUninit2[UL]:
; DIS-NEXT: ...
