; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false --code-model=large -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --relocs --expand-relocs %t.o | FileCheck --check-prefix=RELOC %s
; RUN: llvm-readobj --syms %t.o | FileCheck --check-prefix=SYM %s
; RUN: llvm-objdump -D -r --symbol-description %t.o | FileCheck --check-prefix=DIS %s

@ThreadLocalVarInit = thread_local(localexec) global i64 1, align 8
@VarInit = global i64 87, align 8
@IThreadLocalVarUninit = internal thread_local(localexec) global i64 0, align 8
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
  %2 = load i64, ptr @VarInit, align 8
  %add = add nsw i64 %2, %1
  ret i64 %add
}

; RELOC:      File: {{.*}}aix-tls-le-xcoff-reloc-large32.ll.tmp.o
; RELOC-NEXT: Format: aixcoff-rs6000
; RELOC-NEXT: Arch: powerpc
; RELOC-NEXT: AddressSize: 32bit
; RELOC-NEXT: Relocations [
; RELOC:       Virtual Address: 0x12
; RELOC-NEXT:       Symbol: IThreadLocalVarUninit (17)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 16
; RELOC-NEXT:       Type: R_TOCU (0x30)
; RELOC-NEXT:     }
; RELOC:       Virtual Address: 0x16
; RELOC-NEXT:       Symbol: IThreadLocalVarUninit (17)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 16
; RELOC-NEXT:       Type: R_TOCL (0x31)
; RELOC-NEXT:     }
; RELOC:       Virtual Address: 0x18
; RELOC-NEXT:       Symbol: .__get_tpointer (1)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 26
; RELOC-NEXT:       Type: R_RBA (0x18)
; RELOC-NEXT:     }
; RELOC:       Virtual Address: 0x4E
; RELOC-NEXT:       Symbol: ThreadLocalVarInit (19)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 16
; RELOC-NEXT:       Type: R_TOCU (0x30)
; RELOC-NEXT:     }
; RELOC:       Virtual Address: 0x52
; RELOC-NEXT:       Symbol: ThreadLocalVarInit (19)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 16
; RELOC-NEXT:       Type: R_TOCL (0x31)
; RELOC-NEXT:     }
; RELOC:       Virtual Address: 0x54
; RELOC-NEXT:       Symbol: .__get_tpointer (1)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 26
; RELOC-NEXT:       Type: R_RBA (0x18)
; RELOC-NEXT:     }
; RELOC:       Virtual Address: 0xB0
; RELOC-NEXT:       Symbol: IThreadLocalVarUninit (25)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 32
; RELOC-NEXT:       Type: R_TLS_LE (0x23)
; RELOC-NEXT:     }
; RELOC:     Relocation {
; RELOC-NEXT:       Virtual Address: 0xB4
; RELOC-NEXT:       Symbol: ThreadLocalVarInit (23)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 32
; RELOC-NEXT:       Type: R_TLS_LE (0x23)
; RELOC-NEXT:     }

; SYM:      File: {{.*}}aix-tls-le-xcoff-reloc-large32.ll.tmp.o
; SYM-NEXT: Format: aixcoff-rs6000
; SYM-NEXT: Arch: powerpc
; SYM-NEXT: AddressSize: 32bit
; SYM-NEXT: Symbols [
; SYM:     Index: 1
; SYM-NEXT:     Name: .__get_tpointer
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
; SYM:     Index: 17
; SYM-NEXT:     Name: IThreadLocalVarUninit
; SYM-NEXT:     Value (RelocatableAddress): 0xB0
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 18
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
; SYM:     Index: 19
; SYM-NEXT:     Name: ThreadLocalVarInit
; SYM-NEXT:     Value (RelocatableAddress): 0xB4
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
; SYM:     Index: 23
; SYM-NEXT:     Name: ThreadLocalVarInit
; SYM-NEXT:     Value (RelocatableAddress): 0x0
; SYM-NEXT:     Section: .tdata
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 24
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
; SYM:     Index: 25
; SYM-NEXT:     Name: IThreadLocalVarUninit
; SYM-NEXT:     Value (RelocatableAddress): 0x8
; SYM-NEXT:     Section: .tbss
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 26
; SYM-NEXT:       SectionLen: 8
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 3
; SYM-NEXT:       SymbolType: XTY_CM (0x3)
; SYM-NEXT:       StorageMappingClass: XMC_UL (0x15)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }

; DIS:      {{.*}}aix-tls-le-xcoff-reloc-large32.ll.tmp.o:	file format aixcoff-rs6000
; DIS:      Disassembly of section .text:
; DIS:      00000000 (idx: 5) .storeITLUninit:
; DIS-NEXT:                                       mflr 0
; DIS-NEXT:                                       stwu 1, -32(1)
; DIS-NEXT:                                       stw 0, 40(1)
; DIS-NEXT:                                       mr 5, 3
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                addis 3, 2, 0
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOCU	(idx: 17) IThreadLocalVarUninit[TE]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                lwz 6, 0(3)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOCL	(idx: 17) IThreadLocalVarUninit[TE]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                bla 0
; DIS-NEXT: {{0*}}[[#ADDR]]: R_RBA  (idx: 1)      .__get_tpointer[PR]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                add 3, 3, 6
; DIS-NEXT:                                       stw 4, 4(3)
; DIS-NEXT:                                       stw 5, 0(3)
; DIS-NEXT:                                       addi 1, 1, 32
; DIS-NEXT:                                       lwz 0, 8(1)
; DIS-NEXT:                                       mtlr 0
; DIS-NEXT:                                       blr
; DIS:      00000040 (idx: 7) .loadTLInit:
; DIS-NEXT:                                       mflr 0
; DIS-NEXT:                                       stwu 1, -32(1)
; DIS-NEXT:                                       stw 0, 40(1)
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                addis 3, 2, 0
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOCU       (idx: 19) ThreadLocalVarInit[TE]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                lwz 4, 4(3)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOCL       (idx: 19) ThreadLocalVarInit[TE]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                bla 0
; DIS-NEXT: {{0*}}[[#ADDR]]: R_RBA  (idx: 1)      .__get_tpointer[PR]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                add 3, 3, 4
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                lwz 4, 4(3)
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                lwz 3, 0(3)
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                addis 5, 2, 0
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOCU       (idx: 21) VarInit[TE]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                lwz 5, 8(5)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOCL       (idx: 21) VarInit[TE]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                lwz 6, 4(5)
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                lwz 5, 0(5)
; DIS-NEXT:                                       addc 4, 6, 4
; DIS-NEXT:                                       adde 3, 5, 3
; DIS-NEXT:                                       addi 1, 1, 32
; DIS-NEXT:                                       lwz 0, 8(1)
; DIS-NEXT:                                       mtlr 0
; DIS-NEXT:                                       blr

; DIS:      Disassembly of section .data:
; DIS:      00000090 (idx: 9) VarInit[RW]:
; DIS-NEXT:       90: 00 00 00 00
; DIS-NEXT:       94: 00 00 00 57
; DIS:      00000098 (idx: 11) storeITLUninit[DS]:
; DIS-NEXT:       98: 00 00 00 00
; DIS-NEXT: 00000098:  R_POS        (idx: 5) .storeITLUninit
; DIS-NEXT:       9c: 00 00 00 b0
; DIS-NEXT: 0000009c:  R_POS        (idx: 15) TOC[TC0]
; DIS-NEXT:       a0: 00 00 00 00
; DIS:      000000a4 (idx: 13) loadTLInit[DS]:
; DIS-NEXT:       a4: 00 00 00 40
; DIS-NEXT: 000000a4:  R_POS        (idx: 7) .loadTLInit
; DIS-NEXT:       a8: 00 00 00 b0
; DIS-NEXT: 000000a8:  R_POS        (idx: 15) TOC[TC0]
; DIS-NEXT:       ac: 00 00 00 00
; DIS:      000000b0 (idx: 17) IThreadLocalVarUninit[TE]:
; DIS-NEXT:       b0: 00 00 00 00
; DIS-NEXT: 000000b0:  R_TLS_LE     (idx: 25) IThreadLocalVarUninit[UL]
; DIS:      000000b4 (idx: 19) ThreadLocalVarInit[TE]:
; DIS-NEXT:       b4: 00 00 00 00
; DIS-NEXT: 000000b4:  R_TLS_LE     (idx: 23) ThreadLocalVarInit[TL]
; DIS:      000000b8 (idx: 21) VarInit[TE]:
; DIS-NEXT:       b8: 00 00 00 90
; DIS-NEXT: 000000b8:  R_POS        (idx: 9) VarInit[RW]

; DIS:      Disassembly of section .tdata:
; DIS:      00000000 (idx: 23) ThreadLocalVarInit[TL]:
; DIS-NEXT:        0: 00 00 00 00
; DIS-NEXT:        4: 00 00 00 01

; DIS:      Disassembly of section .tbss:
; DIS:      00000008 (idx: 25) IThreadLocalVarUninit[UL]:
; DIS-NEXT: ...

