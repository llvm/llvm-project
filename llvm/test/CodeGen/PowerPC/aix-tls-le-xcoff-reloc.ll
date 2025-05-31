; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false -data-sections=false -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --relocs --expand-relocs %t.o | FileCheck -D#NFA=2 --check-prefix=RELOC %s
; RUN: llvm-readobj --syms %t.o | FileCheck -D#NFA=2 --check-prefix=SYM %s
; RUN: llvm-objdump -D -r --symbol-description %t.o | FileCheck -D#NFA=2 --check-prefix=DIS %s

@ThreadLocalVarInit = thread_local(localexec) global i32 1, align 4
@VarInit = global i32 87, align 4
@IThreadLocalVarUninit = internal thread_local(localexec) global i32 0, align 4
@IThreadLocalVarUninit2 = internal thread_local(localexec) global i32 0, align 4
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

define signext i32 @loadTLUninit() {
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @IThreadLocalVarUninit)
  store i32 1, ptr %0, align 4
  %1 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @IThreadLocalVarUninit2)
  %2 = load i32, ptr %1, align 4
  %add = add nsw i32 %2, 1
  ret i32 %add
}

; RELOC:      File:
; RELOC-NEXT: Format: aix5coff64-rs6000
; RELOC-NEXT: Arch: powerpc64
; RELOC-NEXT: AddressSize: 64bit
; RELOC-NEXT: Relocations [
; RELOC:       Virtual Address: 0x2
; RELOC-NEXT:       Symbol: IThreadLocalVarUninit ([[#NFA+21]])
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 16
; RELOC-NEXT:       Type: R_TOC (0x3)
; RELOC-NEXT:     }
; RELOC:       Virtual Address: 0x12
; RELOC-NEXT:       Symbol: ThreadLocalVarInit ([[#NFA+23]])
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 16
; RELOC-NEXT:       Type: R_TOC (0x3)
; RELOC-NEXT:     }
; RELOC:       Virtual Address: 0x3E
; RELOC-NEXT:       Symbol: IThreadLocalVarUninit2 ([[#NFA+27]])
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 16
; RELOC-NEXT:       Type: R_TOC (0x3)
; RELOC-NEXT:     }
; RELOC:       Virtual Address: 0xA0
; RELOC-NEXT:       Symbol: IThreadLocalVarUninit ([[#NFA+33]])
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 64
; RELOC-NEXT:       Type: R_TLS_LE (0x23)
; RELOC-NEXT:     }
; RELOC:       Virtual Address: 0xA8
; RELOC-NEXT:       Symbol: ThreadLocalVarInit ([[#NFA+31]])
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 64
; RELOC-NEXT:       Type: R_TLS_LE (0x23)
; RELOC-NEXT:     }
; RELOC:       Virtual Address: 0xB8
; RELOC-NEXT:       Symbol: IThreadLocalVarUninit2 ([[#NFA+35]])
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 64
; RELOC-NEXT:       Type: R_TLS_LE (0x23)
; RELOC-NEXT:     }

; SYM:      File:
; SYM-NEXT: Format: aix5coff64-rs6000
; SYM-NEXT: Arch: powerpc64
; SYM-NEXT: AddressSize: 64bit
; SYM-NEXT: Symbols [
; SYM:     Index: [[#NFA+21]]
; SYM-NEXT:     Name: IThreadLocalVarUninit
; SYM-NEXT:     Value (RelocatableAddress): 0xA0
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#NFA+22]]
; SYM-NEXT:       SectionLen: 8
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 3
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_TC (0x3)
; SYM-NEXT:       Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM:     Index: [[#NFA+23]]
; SYM-NEXT:     Name: ThreadLocalVarInit
; SYM-NEXT:     Value (RelocatableAddress): 0xA8
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#NFA+24]]
; SYM-NEXT:       SectionLen: 8
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 3
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_TC (0x3)
; SYM-NEXT:       Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM:     Index: [[#NFA+27]]
; SYM-NEXT:     Name: IThreadLocalVarUninit2
; SYM-NEXT:     Value (RelocatableAddress): 0xB8
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#NFA+28]]
; SYM-NEXT:       SectionLen: 8
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 3
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_TC (0x3)
; SYM-NEXT:       Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM:     Index: [[#NFA+31]]
; SYM-NEXT:     Name: ThreadLocalVarInit
; SYM-NEXT:     Value (RelocatableAddress): 0x0
; SYM-NEXT:     Section: .tdata
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#NFA+32]]
; SYM-NEXT:       ContainingCsectSymbolIndex: [[#NFA+29]]
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 0
; SYM-NEXT:       SymbolType: XTY_LD (0x2)
; SYM-NEXT:       StorageMappingClass: XMC_TL (0x14)
; SYM-NEXT:       Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM:     Index: [[#NFA+33]]
; SYM-NEXT:     Name: IThreadLocalVarUninit
; SYM-NEXT:     Value (RelocatableAddress): 0x4
; SYM-NEXT:     Section: .tbss
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#NFA+34]]
; SYM-NEXT:       SectionLen: 4
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 2
; SYM-NEXT:       SymbolType: XTY_CM (0x3)
; SYM-NEXT:       StorageMappingClass: XMC_UL (0x15)
; SYM-NEXT:       Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM:     Index: [[#NFA+35]]
; SYM-NEXT:     Name: IThreadLocalVarUninit2
; SYM-NEXT:     Value (RelocatableAddress): 0x8
; SYM-NEXT:     Section: .tbss
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#NFA+36]]
; SYM-NEXT:       SectionLen: 4
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 2
; SYM-NEXT:       SymbolType: XTY_CM (0x3)
; SYM-NEXT:       StorageMappingClass: XMC_UL (0x15)
; SYM-NEXT:       Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:     }
; SYM-NEXT:   }

; DIS:      file format aix5coff64-rs6000
; DIS:      Disassembly of section .text:
; DIS:      0000000000000000 (idx: [[#NFA+3]]) .storeITLUninit:
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               ld 4, 0(2)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOC        (idx: [[#NFA+21]]) IThreadLocalVarUninit[TC]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               stwx 3, 13, 4
; DIS-NEXT:                                      blr
; DIS:      0000000000000010 (idx: [[#NFA+5]]) .loadTLInit:
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               ld 3, 8(2)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOC        (idx: [[#NFA+23]]) ThreadLocalVarInit[TC]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               ld 4, 16(2)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOC        (idx: [[#NFA+25]]) VarInit[TC]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               lwzx 3, 13, 3
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               lwz 4, 0(4)
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               add 3, 4, 3
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               extsw 3, 3
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               blr
; DIS:      0000000000000030 (idx: [[#NFA+7]]) .loadTLUninit:
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               ld 3, 0(2)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOC        (idx: [[#NFA+21]]) IThreadLocalVarUninit[TC]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               li 4, 1
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               stwx 4, 13, 3
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               ld 3, 24(2)
; DIS-NEXT: {{0*}}[[#ADDR + 2]]: R_TOC        (idx: [[#NFA+27]]) IThreadLocalVarUninit2[TC]
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               lwzx 3, 13, 3
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               addi 3, 3, 1
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               extsw 3, 3
; DIS-NEXT: [[#%x, ADDR:]]: {{.*}}               blr

; DIS:      Disassembly of section .data:
; DIS:      0000000000000050 (idx: [[#NFA+11]]) VarInit:
; DIS-NEXT:       50: 00 00 00 57
; DIS:      0000000000000058 (idx: [[#NFA+13]]) storeITLUninit[DS]:
; DIS-NEXT:       58: 00 00 00 00
; DIS-NEXT: 0000000000000058:  R_POS        (idx: [[#NFA+3]]) .storeITLUninit
; DIS-NEXT:       5c: 00 00 00 00
; DIS-NEXT:       60: 00 00 00 00
; DIS-NEXT: 0000000000000060:  R_POS        (idx: [[#NFA+19]]) TOC[TC0]
; DIS-NEXT:       64: 00 00 00 a0
; DIS:      0000000000000070 (idx: [[#NFA+15]]) loadTLInit[DS]:
; DIS-NEXT:       70: 00 00 00 00
; DIS-NEXT: 0000000000000070:  R_POS        (idx: [[#NFA+5]]) .loadTLInit
; DIS-NEXT:       74: 00 00 00 10
; DIS-NEXT:       78: 00 00 00 00
; DIS-NEXT: 0000000000000078:  R_POS        (idx: [[#NFA+19]]) TOC[TC0]
; DIS-NEXT:       7c: 00 00 00 a0
; DIS:      0000000000000088 (idx: [[#NFA+17]]) loadTLUninit[DS]:
; DIS-NEXT:       88: 00 00 00 00
; DIS-NEXT: 0000000000000088:  R_POS        (idx: [[#NFA+7]]) .loadTLUninit
; DIS-NEXT:       8c: 00 00 00 30
; DIS-NEXT:       90: 00 00 00 00
; DIS-NEXT: 0000000000000090:  R_POS        (idx: [[#NFA+19]]) TOC[TC0]
; DIS-NEXT:       94: 00 00 00 a0
; DIS:      00000000000000a0 (idx: [[#NFA+21]]) IThreadLocalVarUninit[TC]:
; DIS-NEXT:       a0: 00 00 00 00
; DIS-NEXT: 00000000000000a0:  R_TLS_LE     (idx: [[#NFA+33]]) IThreadLocalVarUninit[UL]
; DIS-NEXT:       a4: 00 00 00 04
; DIS:      00000000000000a8 (idx: [[#NFA+23]]) ThreadLocalVarInit[TC]:
; DIS-NEXT:       a8: 00 00 00 00
; DIS-NEXT: 00000000000000a8:  R_TLS_LE     (idx: [[#NFA+31]]) ThreadLocalVarInit
; DIS-NEXT:       ac: 00 00 00 00
; DIS:      00000000000000b0 (idx: [[#NFA+25]]) VarInit[TC]:
; DIS-NEXT:       b0: 00 00 00 00
; DIS-NEXT: 00000000000000b0:  R_POS        (idx: [[#NFA+11]]) VarInit
; DIS-NEXT:       b4: 00 00 00 50
; DIS:      00000000000000b8 (idx: [[#NFA+27]]) IThreadLocalVarUninit2[TC]:
; DIS-NEXT:       b8: 00 00 00 00
; DIS-NEXT: 00000000000000b8:  R_TLS_LE     (idx: [[#NFA+35]]) IThreadLocalVarUninit2[UL]
; DIS-NEXT:       bc: 00 00 00 08

; DIS:      Disassembly of section .tdata:
; DIS:      0000000000000000 (idx: [[#NFA+31]]) ThreadLocalVarInit:
; DIS-NEXT:        0: 00 00 00 01

; DIS:      Disassembly of section .tbss:
; DIS:      0000000000000004 (idx: [[#NFA+33]]) IThreadLocalVarUninit[UL]:
; DIS-NEXT: ...
; DIS:      0000000000000008 (idx: [[#NFA+35]]) IThreadLocalVarUninit2[UL]:
; DIS-NEXT: ...

