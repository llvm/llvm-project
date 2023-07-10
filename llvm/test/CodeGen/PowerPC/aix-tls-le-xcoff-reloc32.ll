; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false -data-sections=false -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --relocs --expand-relocs %t.o | FileCheck --check-prefix=RELOC %s
; RUN: llvm-readobj --syms %t.o | FileCheck --check-prefix=SYM %s
; RUN: llvm-objdump -D -r --symbol-description %t.o | FileCheck --check-prefix=DIS %s

@ThreadLocalVarInit = thread_local(localexec) global i32 1, align 4
@VarInit = global i32 87, align 4
@IThreadLocalVarUninit = internal thread_local(localexec) global i32 0, align 4
declare nonnull ptr @llvm.threadlocal.address.p0(ptr nonnull)

define void @storeITLUninit(i32 noundef signext %x) {
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @IThreadLocalVarUninit)
  store i32 %x, ptr %0, align 4
  ret void
}

define signext i32 @loadTLInit() {
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @ThreadLocalVarInit)
  %1 = load i32, ptr %0, align 4
  %2 = load i32, ptr @VarInit, align 4
  %add = add nsw i32 %2, %1
  ret i32 %add
}

; RELOC:      File: {{.*}}aix-tls-le-xcoff-reloc32.ll.tmp.o
; RELOC-NEXT: Format: aixcoff-rs6000
; RELOC-NEXT: Arch: powerpc
; RELOC-NEXT: AddressSize: 32bit
; RELOC-NEXT: Relocations [
; RELOC:       Virtual Address: 0xA
; RELOC-NEXT:       Symbol: IThreadLocalVarUninit (19)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 16
; RELOC-NEXT:       Type: R_TOC (0x3)
; RELOC-NEXT:     }
; RELOC:       Virtual Address: 0x10
; RELOC-NEXT:       Symbol: .__get_tpointer (1)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 26
; RELOC-NEXT:       Type: R_RBA (0x18)
; RELOC-NEXT:     }
; RELOC:       Virtual Address: 0x3A
; RELOC-NEXT:       Symbol: ThreadLocalVarInit (21)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 16
; RELOC-NEXT:       Type: R_TOC (0x3)
; RELOC-NEXT:     }
; RELOC:       Virtual Address: 0x44
; RELOC-NEXT:       Symbol: .__get_tpointer (1)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 26
; RELOC-NEXT:       Type: R_RBA (0x18)
; RELOC-NEXT:     }
; RELOC:       Virtual Address: 0x80
; RELOC-NEXT:       Symbol: IThreadLocalVarUninit (29)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 32
; RELOC-NEXT:       Type: R_TLS_LE (0x23)
; RELOC-NEXT:     }
; RELOC:       Virtual Address: 0x84
; RELOC-NEXT:       Symbol: ThreadLocalVarInit (27)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 32
; RELOC-NEXT:       Type: R_TLS_LE (0x23)
; RELOC-NEXT:     }

; SYM:      File: {{.*}}aix-tls-le-xcoff-reloc32.ll.tmp.o
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
; SYM:     Index: 19
; SYM-NEXT:     Name: IThreadLocalVarUninit
; SYM-NEXT:     Value (RelocatableAddress): 0x80
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
; SYM-NEXT:       StorageMappingClass: XMC_TC (0x3)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM:     Index: 21
; SYM-NEXT:     Name: ThreadLocalVarInit
; SYM-NEXT:     Value (RelocatableAddress): 0x84
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
; SYM:     Index: 27
; SYM-NEXT:     Name: ThreadLocalVarInit
; SYM-NEXT:     Value (RelocatableAddress): 0x0
; SYM-NEXT:     Section: .tdata
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 28
; SYM-NEXT:       ContainingCsectSymbolIndex: 25
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 0
; SYM-NEXT:       SymbolType: XTY_LD (0x2)
; SYM-NEXT:       StorageMappingClass: XMC_TL (0x14)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM:     Index: 29
; SYM-NEXT:     Name: IThreadLocalVarUninit
; SYM-NEXT:     Value (RelocatableAddress): 0x4
; SYM-NEXT:     Section: .tbss
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 30
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

; DIS:      {{.*}}aix-tls-le-xcoff-reloc32.ll.tmp.o:	file format aixcoff-rs6000
; DIS:      Disassembly of section .text:
; DIS:      00000000 (idx: 5) .storeITLUninit:
; DIS-NEXT:                                      mflr 0
; DIS-NEXT:                                      stwu 1, -32(1)
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               lwz 5, 0(2)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOC        (idx: 19) IThreadLocalVarUninit[TC]
; DIS-NEXT:                                      mr 4, 3
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               bla 0
; DIS-NEXT: {{0*}}[[#ADDR]]: R_RBA (idx: 1)      .__get_tpointer[PR]
; DIS-NEXT:                                      stw 0, 40(1)
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               stwx 4, 3, 5
; DIS-NEXT:                                      addi 1, 1, 32
; DIS-NEXT:                                      lwz 0, 8(1)
; DIS-NEXT:                                      mtlr 0
; DIS-NEXT:                                      blr
; DIS:      00000030 (idx: 7) .loadTLInit:
; DIS-NEXT:                                      mflr 0
; DIS-NEXT:                                      stwu 1, -32(1)
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               lwz 4, 4(2)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOC        (idx: 21) ThreadLocalVarInit[TC]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               lwz 5, 8(2)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOC        (idx: 23) VarInit[TC]
; DIS-NEXT:                                      stw 0, 40(1)
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               bla 0
; DIS-NEXT: {{0*}}[[#ADDR]]: R_RBA (idx: 1)      .__get_tpointer[PR]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               lwzx 3, 3, 4
; DIS-NEXT:                                      lwz 4, 0(5)
; DIS-NEXT:                                      add 3, 4, 3
; DIS-NEXT:                                      addi 1, 1, 32
; DIS-NEXT:                                      lwz 0, 8(1)
; DIS-NEXT:                                      mtlr 0
; DIS-NEXT:                                      blr

; DIS:      Disassembly of section .data:
; DIS:      00000064 (idx: 11) VarInit:
; DIS-NEXT:       64: 00 00 00 57
; DIS:      00000068 (idx: 13) storeITLUninit[DS]:
; DIS-NEXT:       68: 00 00 00 00
; DIS-NEXT: 00000068:  R_POS        (idx: 5) .storeITLUninit
; DIS-NEXT:       6c: 00 00 00 80
; DIS-NEXT: 0000006c:  R_POS        (idx: 17) TOC[TC0]
; DIS-NEXT:       70: 00 00 00 00
; DIS:      00000074 (idx: 15) loadTLInit[DS]:
; DIS-NEXT:       74: 00 00 00 30
; DIS-NEXT: 00000074:  R_POS        (idx: 7) .loadTLInit
; DIS-NEXT:       78: 00 00 00 80
; DIS-NEXT: 00000078:  R_POS        (idx: 17) TOC[TC0]
; DIS-NEXT:       7c: 00 00 00 00
; DIS:      00000080 (idx: 19) IThreadLocalVarUninit[TC]:
; DIS-NEXT:       80: 00 00 00 00
; DIS-NEXT: 00000080:  R_TLS_LE     (idx: 29) IThreadLocalVarUninit[UL]
; DIS:      00000084 (idx: 21) ThreadLocalVarInit[TC]:
; DIS-NEXT:       84: 00 00 00 00
; DIS-NEXT: 00000084:  R_TLS_LE     (idx: 27) ThreadLocalVarInit
; DIS:      00000088 (idx: 23) VarInit[TC]:
; DIS-NEXT:       88: 00 00 00 64
; DIS-NEXT: 00000088:  R_POS        (idx: 11) VarInit

; DIS:      Disassembly of section .tdata:
; DIS:      00000000 (idx: 27) ThreadLocalVarInit:
; DIS-NEXT:        0: 00 00 00 01

; DIS:      Disassembly of section .tbss:
; DIS:      00000004 (idx: 29) IThreadLocalVarUninit[UL]:
; DIS-NEXT: ...

