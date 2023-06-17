; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff \
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

; RELOC:      File: {{.*}}aix-tls-le-xcoff-reloc-large.ll.tmp.o
; RELOC-NEXT: Format: aix5coff64-rs6000
; RELOC-NEXT: Arch: powerpc64
; RELOC-NEXT: AddressSize: 64bit
; RELOC-NEXT: Relocations [
; RELOC:       Virtual Address: 0x2
; RELOC-NEXT:       Symbol: IThreadLocalVarUninit (15)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 16
; RELOC-NEXT:       Type: R_TOCU (0x30)
; RELOC-NEXT:     }
; RELOC:       Virtual Address: 0x6
; RELOC-NEXT:       Symbol: IThreadLocalVarUninit (15)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 16
; RELOC-NEXT:       Type: R_TOCL (0x31)
; RELOC-NEXT:     }
; RELOC:       Virtual Address: 0x22
; RELOC-NEXT:       Symbol: ThreadLocalVarInit (17)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 16
; RELOC-NEXT:       Type: R_TOCU (0x30)
; RELOC-NEXT:     }
; RELOC:       Virtual Address: 0x2A
; RELOC-NEXT:       Symbol: ThreadLocalVarInit (17)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 16
; RELOC-NEXT:       Type: R_TOCL (0x31)
; RELOC-NEXT:     }
; RELOC:       Virtual Address: 0x80
; RELOC-NEXT:       Symbol: IThreadLocalVarUninit (23)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 64
; RELOC-NEXT:       Type: R_TLS_LE (0x23)
; RELOC-NEXT:     }
; RELOC:       Virtual Address: 0x88
; RELOC-NEXT:       Symbol: ThreadLocalVarInit (21)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 64
; RELOC-NEXT:       Type: R_TLS_LE (0x23)
; RELOC-NEXT:     }

; SYM:      File: {{.*}}aix-tls-le-xcoff-reloc-large.ll.tmp.o
; SYM-NEXT: Format: aix5coff64-rs6000
; SYM-NEXT: Arch: powerpc64
; SYM-NEXT: AddressSize: 64bit
; SYM-NEXT: Symbols [
; SYM:     Index: 15
; SYM-NEXT:     Name: IThreadLocalVarUninit
; SYM-NEXT:     Value (RelocatableAddress): 0x80
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 16
; SYM-NEXT:       SectionLen: 8
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 3
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_TE (0x16)
; SYM-NEXT:       Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM:     Index: 17
; SYM-NEXT:     Name: ThreadLocalVarInit
; SYM-NEXT:     Value (RelocatableAddress): 0x88
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 18
; SYM-NEXT:       SectionLen: 8
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 3
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_TE (0x16)
; SYM-NEXT:       Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM:     Index: 21
; SYM-NEXT:     Name: ThreadLocalVarInit
; SYM-NEXT:     Value (RelocatableAddress): 0x0
; SYM-NEXT:     Section: .tdata
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 22
; SYM-NEXT:       SectionLen: 8
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 3
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_TL (0x14)
; SYM-NEXT:       Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM:     Index: 23
; SYM-NEXT:     Name: IThreadLocalVarUninit
; SYM-NEXT:     Value (RelocatableAddress): 0x8
; SYM-NEXT:     Section: .tbss
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 24
; SYM-NEXT:       SectionLen: 8
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 3
; SYM-NEXT:       SymbolType: XTY_CM (0x3)
; SYM-NEXT:       StorageMappingClass: XMC_UL (0x15)
; SYM-NEXT:       Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:     }
; SYM-NEXT:   }

; DIS:      {{.*}}aix-tls-le-xcoff-reloc-large.ll.tmp.o:	file format aix5coff64-rs6000
; DIS:      Disassembly of section .text:
; DIS:      0000000000000000 (idx: 3) .storeITLUninit:
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                addis 4, 2, 0
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOCU	(idx: 15) IThreadLocalVarUninit[TE]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                ld 4, 0(4)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOCL	(idx: 15) IThreadLocalVarUninit[TE]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                add 4, 13, 4
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                std 3, 0(4)
; DIS-NEXT:                                       blr
; DIS:      0000000000000020 (idx: 5) .loadTLInit:
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                addis 3, 2, 0
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOCU       (idx: 17) ThreadLocalVarInit[TE]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                addis 4, 2, 0
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOCU       (idx: 19) VarInit[TE]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                ld 3, 8(3)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOCL       (idx: 17) ThreadLocalVarInit[TE]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                ld 4, 16(4)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOCL       (idx: 19) VarInit[TE]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                add 3, 13, 3
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                ld 4, 0(4)
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                ld 3, 0(3)
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                add 3, 4, 3
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}                blr

; DIS:      Disassembly of section .data:
; DIS:      0000000000000048 (idx: 7) VarInit[RW]:
; DIS-NEXT:       48: 00 00 00 00
; DIS-NEXT:       4c: 00 00 00 57
; DIS:      0000000000000050 (idx: 9) storeITLUninit[DS]:
; DIS-NEXT:       50: 00 00 00 00
; DIS-NEXT: 0000000000000050:  R_POS        (idx: 3) .storeITLUninit
; DIS-NEXT:       54: 00 00 00 00
; DIS-NEXT:       58: 00 00 00 00
; DIS-NEXT: 0000000000000058:  R_POS        (idx: 13) TOC[TC0]
; DIS-NEXT:       5c: 00 00 00 80
; DIS:      0000000000000068 (idx: 11) loadTLInit[DS]:
; DIS-NEXT:       68: 00 00 00 00
; DIS-NEXT: 0000000000000068:  R_POS        (idx: 5) .loadTLInit
; DIS-NEXT:       6c: 00 00 00 20
; DIS-NEXT:       70: 00 00 00 00
; DIS-NEXT: 0000000000000070:  R_POS        (idx: 13) TOC[TC0]
; DIS-NEXT:       74: 00 00 00 80
; DIS:      0000000000000080 (idx: 15) IThreadLocalVarUninit[TE]:
; DIS-NEXT:       80: 00 00 00 00
; DIS-NEXT: 0000000000000080:  R_TLS_LE     (idx: 23) IThreadLocalVarUninit[UL]
; DIS-NEXT:       84: 00 00 00 00
; DIS:      0000000000000088 (idx: 17) ThreadLocalVarInit[TE]:
; DIS-NEXT:       88: 00 00 00 00
; DIS-NEXT: 0000000000000088:  R_TLS_LE     (idx: 21) ThreadLocalVarInit[TL]
; DIS-NEXT:       8c: 00 00 00 00
; DIS:      0000000000000090 (idx: 19) VarInit[TE]:
; DIS-NEXT:       90: 00 00 00 00
; DIS-NEXT: 0000000000000090:  R_POS        (idx: 7) VarInit[RW]
; DIS-NEXT:       94: 00 00 00 48

; DIS:      Disassembly of section .tdata:
; DIS:      0000000000000000 (idx: 21) ThreadLocalVarInit[TL]:
; DIS-NEXT:        0: 00 00 00 00
; DIS-NEXT:        4: 00 00 00 01

; DIS:      Disassembly of section .tbss:
; DIS:      0000000000000008 (idx: 23) IThreadLocalVarUninit[UL]:
; DIS-NEXT: ...

