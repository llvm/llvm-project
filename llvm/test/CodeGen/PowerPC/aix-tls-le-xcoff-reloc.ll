; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff \
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

; RELOC:      File: {{.*}}aix-tls-le-xcoff-reloc.ll.tmp.o
; RELOC-NEXT: Format: aix5coff64-rs6000
; RELOC-NEXT: Arch: powerpc64
; RELOC-NEXT: AddressSize: 64bit
; RELOC-NEXT: Relocations [
; RELOC:       Virtual Address: 0x2
; RELOC-NEXT:       Symbol: IThreadLocalVarUninit (17)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 16
; RELOC-NEXT:       Type: R_TOC (0x3)
; RELOC-NEXT:     }
; RELOC:       Virtual Address: 0x12
; RELOC-NEXT:       Symbol: ThreadLocalVarInit (19)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 16
; RELOC-NEXT:       Type: R_TOC (0x3)
; RELOC-NEXT:     }
; RELOC:       Virtual Address: 0x68
; RELOC-NEXT:       Symbol: IThreadLocalVarUninit (27)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 64
; RELOC-NEXT:       Type: R_TLS_LE (0x23)
; RELOC-NEXT:     }
; RELOC:       Virtual Address: 0x70
; RELOC-NEXT:       Symbol: ThreadLocalVarInit (25)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 64
; RELOC-NEXT:       Type: R_TLS_LE (0x23)
; RELOC-NEXT:     }

; SYM:      File: {{.*}}aix-tls-le-xcoff-reloc.ll.tmp.o
; SYM-NEXT: Format: aix5coff64-rs6000
; SYM-NEXT: Arch: powerpc64
; SYM-NEXT: AddressSize: 64bit
; SYM-NEXT: Symbols [
; SYM:     Index: 17
; SYM-NEXT:     Name: IThreadLocalVarUninit
; SYM-NEXT:     Value (RelocatableAddress): 0x68
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
; SYM-NEXT:       StorageMappingClass: XMC_TC (0x3)
; SYM-NEXT:       Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM:     Index: 19
; SYM-NEXT:     Name: ThreadLocalVarInit
; SYM-NEXT:     Value (RelocatableAddress): 0x70
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 20
; SYM-NEXT:       SectionLen: 8
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 3
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_TC (0x3)
; SYM-NEXT:       Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM:     Index: 25
; SYM-NEXT:     Name: ThreadLocalVarInit
; SYM-NEXT:     Value (RelocatableAddress): 0x0
; SYM-NEXT:     Section: .tdata
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 26
; SYM-NEXT:       ContainingCsectSymbolIndex: 23
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 0
; SYM-NEXT:       SymbolType: XTY_LD (0x2)
; SYM-NEXT:       StorageMappingClass: XMC_TL (0x14)
; SYM-NEXT:       Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM:     Index: 27
; SYM-NEXT:     Name: IThreadLocalVarUninit
; SYM-NEXT:     Value (RelocatableAddress): 0x4
; SYM-NEXT:     Section: .tbss
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 28
; SYM-NEXT:       SectionLen: 4
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 2
; SYM-NEXT:       SymbolType: XTY_CM (0x3)
; SYM-NEXT:       StorageMappingClass: XMC_UL (0x15)
; SYM-NEXT:       Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:     }
; SYM-NEXT:   }

; DIS:      {{.*}}aix-tls-le-xcoff-reloc.ll.tmp.o:	file format aix5coff64-rs6000
; DIS:      Disassembly of section .text:
; DIS:      0000000000000000 (idx: 3) .storeITLUninit:
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               ld 4, 0(2)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOC        (idx: 17) IThreadLocalVarUninit[TC]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               add 4, 13, 4
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               stw 3, 0(4)
; DIS-NEXT:                                      blr
; DIS:      0000000000000010 (idx: 5) .loadTLInit:
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               ld 3, 8(2)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOC        (idx: 19) ThreadLocalVarInit[TC]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               ld 4, 16(2)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOC        (idx: 21) VarInit[TC]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               add 3, 13, 3
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               lwz 4, 0(4)
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               lwz 3, 0(3)
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               add 3, 4, 3
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               extsw 3, 3
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               blr

; DIS:      Disassembly of section .data:
; DIS:      0000000000000030 (idx: 9) VarInit:
; DIS-NEXT:       30: 00 00 00 57
; DIS:      0000000000000038 (idx: 11) storeITLUninit[DS]:
; DIS-NEXT:       8: 00 00 00 00
; DIS-NEXT: 0000000000000038:  R_POS        (idx: 3) .storeITLUninit
; DIS-NEXT:       3c: 00 00 00 00
; DIS-NEXT:       40: 00 00 00 00
; DIS-NEXT: 0000000000000040:  R_POS        (idx: 15) TOC[TC0]
; DIS-NEXT:       44: 00 00 00 68
; DIS:      0000000000000050 (idx: 13) loadTLInit[DS]:
; DIS-NEXT:       50: 00 00 00 00
; DIS-NEXT: 0000000000000050:  R_POS        (idx: 5) .loadTLInit
; DIS-NEXT:       54: 00 00 00 10
; DIS-NEXT:       58: 00 00 00 00
; DIS-NEXT: 0000000000000058:  R_POS        (idx: 15) TOC[TC0]
; DIS-NEXT:       5c: 00 00 00 68
; DIS:      0000000000000068 (idx: 17) IThreadLocalVarUninit[TC]:
; DIS-NEXT:       68: 00 00 00 00
; DIS-NEXT: 0000000000000068:  R_TLS_LE     (idx: 27) IThreadLocalVarUninit[UL]
; DIS:      0000000000000070 (idx: 19) ThreadLocalVarInit[TC]:
; DIS-NEXT:       70: 00 00 00 00
; DIS-NEXT: 0000000000000070:  R_TLS_LE     (idx: 25) ThreadLocalVarInit
; DIS:      0000000000000078 (idx: 21) VarInit[TC]:
; DIS-NEXT:       78: 00 00 00 00
; DIS-NEXT: 0000000000000078:  R_POS        (idx: 9) VarInit

; DIS:      Disassembly of section .tdata:
; DIS:      0000000000000000 (idx: 25) ThreadLocalVarInit:
; DIS-NEXT:        0: 00 00 00 01

; DIS:      Disassembly of section .tbss:
; DIS:      0000000000000004 (idx: 27) IThreadLocalVarUninit[UL]:
; DIS-NEXT: ...

