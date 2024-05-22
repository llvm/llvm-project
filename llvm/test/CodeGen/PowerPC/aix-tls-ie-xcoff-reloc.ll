; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple=powerpc64-ibm-aix-xcoff \
; RUN:   -xcoff-traceback-table=false -data-sections=false -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --relocs --expand-relocs %t.o | FileCheck --check-prefix=REL64 %s
; RUN: llvm-readobj --syms %t.o | FileCheck --check-prefix=SYM64 %s
; RUN: llvm-objdump -D -r --symbol-description %t.o | FileCheck --check-prefix=DIS64 %s

; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple=powerpc64-ibm-aix-xcoff -code-model=small \
; RUN:   -xcoff-traceback-table=false -data-sections=false -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --relocs --expand-relocs %t.o | FileCheck --check-prefix=REL64 %s
; RUN: llvm-readobj --syms %t.o | FileCheck --check-prefix=SYM64 %s
; RUN: llvm-objdump -D -r --symbol-description %t.o | FileCheck --check-prefix=DIS64 %s

; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple=powerpc-ibm-aix-xcoff \
; RUN:   -xcoff-traceback-table=false -data-sections=false -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --relocs --expand-relocs %t.o | FileCheck --check-prefix=REL32 %s
; RUN: llvm-readobj --syms %t.o | FileCheck --check-prefix=SYM32 %s
; RUN: llvm-objdump -D -r --symbol-description %t.o | FileCheck --check-prefix=DIS32 %s

; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple=powerpc-ibm-aix-xcoff -code-model=small \
; RUN:   -xcoff-traceback-table=false -data-sections=false -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --relocs --expand-relocs %t.o | FileCheck --check-prefix=REL32 %s
; RUN: llvm-readobj --syms %t.o | FileCheck --check-prefix=SYM32 %s
; RUN: llvm-objdump -D -r --symbol-description %t.o | FileCheck --check-prefix=DIS32 %s

@global_int_nonzero = thread_local(initialexec) global i32 1, align 4
@intern_int_zero = internal thread_local(initialexec) global i32 0, align 4

define void @store_intern_int_zero(i32 noundef signext %i) {
entry:
  %addr = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @intern_int_zero)
  store i32 %i, ptr %addr, align 4
  ret void
}

define signext i32 @load_global_int_nonzero() {
entry:
  %addr = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @global_int_nonzero)
  %val = load i32, ptr %addr, align 4
  ret i32 %val
}

define signext i32 @load_intern_int_zero() {
entry:
  %addr = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @intern_int_zero)
  %val = load i32, ptr %addr, align 4
  ret i32 %val
}

declare nonnull ptr @llvm.threadlocal.address.p0(ptr nonnull)

; REL64:      File: {{.*}}aix-tls-ie-xcoff-reloc.ll.tmp.o
; REL64-NEXT: Format: aix5coff64-rs6000
; REL64-NEXT: Arch: powerpc64
; REL64-NEXT: AddressSize: 64bit
; REL64-NEXT: Relocations [
; REL64:       Virtual Address: 0x2
; REL64-NEXT:       Symbol: intern_int_zero (17)
; REL64-NEXT:       IsSigned: No
; REL64-NEXT:       FixupBitValue: 0
; REL64-NEXT:       Length: 16
; REL64-NEXT:       Type: R_TOC (0x3)
; REL64-NEXT:     }
; REL64:       Virtual Address: 0x12
; REL64-NEXT:       Symbol: global_int_nonzero (19)
; REL64-NEXT:       IsSigned: No
; REL64-NEXT:       FixupBitValue: 0
; REL64-NEXT:       Length: 16
; REL64-NEXT:       Type: R_TOC (0x3)
; REL64-NEXT:     }
; REL64:       Virtual Address: 0x22
; REL64-NEXT:       Symbol: intern_int_zero (17)
; REL64-NEXT:       IsSigned: No
; REL64-NEXT:       FixupBitValue: 0
; REL64-NEXT:       Length: 16
; REL64-NEXT:       Type: R_TOC (0x3)
; REL64-NEXT:     }
; REL64:       Virtual Address: 0x78
; REL64-NEXT:       Symbol: intern_int_zero (25)
; REL64-NEXT:       IsSigned: No
; REL64-NEXT:       FixupBitValue: 0
; REL64-NEXT:       Length: 64
; REL64-NEXT:       Type: R_TLS_IE (0x21)
; REL64-NEXT:     }
; REL64:       Virtual Address: 0x80
; REL64-NEXT:       Symbol: global_int_nonzero (23)
; REL64-NEXT:       IsSigned: No
; REL64-NEXT:       FixupBitValue: 0
; REL64-NEXT:       Length: 64
; REL64-NEXT:       Type: R_TLS_IE (0x21)
; REL64-NEXT:     }

; SYM64:      File: {{.*}}aix-tls-ie-xcoff-reloc.ll.tmp.o
; SYM64-NEXT: Format: aix5coff64-rs6000
; SYM64-NEXT: Arch: powerpc64
; SYM64-NEXT: AddressSize: 64bit
; SYM64-NEXT: Symbols [
; SYM64:     Index: 17
; SYM64-NEXT:     Name: intern_int_zero
; SYM64-NEXT:     Value (RelocatableAddress): 0x78
; SYM64-NEXT:     Section: .data
; SYM64-NEXT:     Type: 0x0
; SYM64-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM64-NEXT:     NumberOfAuxEntries: 1
; SYM64-NEXT:     CSECT Auxiliary Entry {
; SYM64-NEXT:       Index: 18
; SYM64-NEXT:       SectionLen: 8
; SYM64-NEXT:       ParameterHashIndex: 0x0
; SYM64-NEXT:       TypeChkSectNum: 0x0
; SYM64-NEXT:       SymbolAlignmentLog2: 3
; SYM64-NEXT:       SymbolType: XTY_SD (0x1)
; SYM64-NEXT:       StorageMappingClass: XMC_TC (0x3)
; SYM64-NEXT:       Auxiliary Type: AUX_CSECT (0xFB)
; SYM64-NEXT:     }
; SYM64-NEXT:   }
; SYM64:     Index: 19
; SYM64-NEXT:     Name: global_int_nonzero
; SYM64-NEXT:     Value (RelocatableAddress): 0x80
; SYM64-NEXT:     Section: .data
; SYM64-NEXT:     Type: 0x0
; SYM64-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM64-NEXT:     NumberOfAuxEntries: 1
; SYM64-NEXT:     CSECT Auxiliary Entry {
; SYM64-NEXT:       Index: 20
; SYM64-NEXT:       SectionLen: 8
; SYM64-NEXT:       ParameterHashIndex: 0x0
; SYM64-NEXT:       TypeChkSectNum: 0x0
; SYM64-NEXT:       SymbolAlignmentLog2: 3
; SYM64-NEXT:       SymbolType: XTY_SD (0x1)
; SYM64-NEXT:       StorageMappingClass: XMC_TC (0x3)
; SYM64-NEXT:       Auxiliary Type: AUX_CSECT (0xFB)
; SYM64-NEXT:     }
; SYM64-NEXT:   }
; SYM64:     Index: 23
; SYM64-NEXT:     Name: global_int_nonzero
; SYM64-NEXT:     Value (RelocatableAddress): 0x0
; SYM64-NEXT:     Section: .tdata
; SYM64-NEXT:     Type: 0x0
; SYM64-NEXT:     StorageClass: C_EXT (0x2)
; SYM64-NEXT:     NumberOfAuxEntries: 1
; SYM64-NEXT:     CSECT Auxiliary Entry {
; SYM64-NEXT:       Index: 24
; SYM64-NEXT:       ContainingCsectSymbolIndex: 21
; SYM64-NEXT:       ParameterHashIndex: 0x0
; SYM64-NEXT:       TypeChkSectNum: 0x0
; SYM64-NEXT:       SymbolAlignmentLog2: 0
; SYM64-NEXT:       SymbolType: XTY_LD (0x2)
; SYM64-NEXT:       StorageMappingClass: XMC_TL (0x14)
; SYM64-NEXT:       Auxiliary Type: AUX_CSECT (0xFB)
; SYM64-NEXT:     }
; SYM64-NEXT:   }
; SYM64:     Index: 25
; SYM64-NEXT:     Name: intern_int_zero
; SYM64-NEXT:     Value (RelocatableAddress): 0x4
; SYM64-NEXT:     Section: .tbss
; SYM64-NEXT:     Type: 0x0
; SYM64-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM64-NEXT:     NumberOfAuxEntries: 1
; SYM64-NEXT:     CSECT Auxiliary Entry {
; SYM64-NEXT:       Index: 26
; SYM64-NEXT:       SectionLen: 4
; SYM64-NEXT:       ParameterHashIndex: 0x0
; SYM64-NEXT:       TypeChkSectNum: 0x0
; SYM64-NEXT:       SymbolAlignmentLog2: 2
; SYM64-NEXT:       SymbolType: XTY_CM (0x3)
; SYM64-NEXT:       StorageMappingClass: XMC_UL (0x15)
; SYM64-NEXT:       Auxiliary Type: AUX_CSECT (0xFB)
; SYM64-NEXT:     }
; SYM64-NEXT:   }

; DIS64: {{.*}}aix-tls-ie-xcoff-reloc.ll.tmp.o:	file format aix5coff64-rs6000
; DIS64: Disassembly of section .text:
; DIS64: (idx: 3) .store_intern_int_zero:
; DIS64-NEXT: ld 4, 0(2)
; DIS64-NEXT: (idx: 17) intern_int_zero[TC]
; DIS64-NEXT: stwx 3, 13, 4
; DIS64-NEXT: blr
; DIS64: (idx: 5) .load_global_int_nonzero:
; DIS64-NEXT: ld 3, 8(2)
; DIS64-NEXT: (idx: 19) global_int_nonzero[TC]
; DIS64-NEXT: lwax 3, 13, 3
; DIS64-NEXT: blr
; DIS64: (idx: 7) .load_intern_int_zero:
; DIS64-NEXT: ld 3, 0(2)
; DIS64-NEXT: (idx: 17) intern_int_zero[TC]
; DIS64-NEXT: lwax 3, 13, 3
; DIS64-NEXT: blr

; DIS64: Disassembly of section .data:
; DIS64: (idx: 9) store_intern_int_zero[DS]:
; DIS64: R_POS        (idx: 3) .store_intern_int_zero
; DIS64: R_POS        (idx: 15) TOC[TC0]
; DIS64: (idx: 11) load_global_int_nonzero[DS]:
; DIS64: R_POS        (idx: 5) .load_global_int_nonzero
; DIS64: R_POS        (idx: 15) TOC[TC0]
; DIS64: (idx: 13) load_intern_int_zero[DS]:
; DIS64: R_POS        (idx: 7) .load_intern_int_zero
; DIS64: R_POS        (idx: 15) TOC[TC0]
; DIS64: (idx: 17) intern_int_zero[TC]:
; DIS64: R_TLS_IE     (idx: 25) intern_int_zero[UL]
; DIS64: (idx: 19) global_int_nonzero[TC]:
; DIS64: R_TLS_IE     (idx: 23) global_int_nonzero

; DIS64: Disassembly of section .tdata:
; DIS64: (idx: 23) global_int_nonzero:

; DIS64: Disassembly of section .tbss:
; DIS64: (idx: 25) intern_int_zero[UL]:

; REL32:      File: {{.*}}aix-tls-ie-xcoff-reloc.ll.tmp.o
; REL32-NEXT: Format: aixcoff-rs6000
; REL32-NEXT: Arch: powerpc
; REL32-NEXT: AddressSize: 32bit
; REL32-NEXT: Relocations [
; REL32:       Virtual Address: 0xA
; REL32-NEXT:       Symbol: intern_int_zero (19)
; REL32-NEXT:       IsSigned: No
; REL32-NEXT:       FixupBitValue: 0
; REL32-NEXT:       Length: 16
; REL32-NEXT:       Type: R_TOC (0x3)
; REL32-NEXT:     }
; REL32:       Virtual Address: 0x10
; REL32-NEXT:       Symbol: .__get_tpointer (1)
; REL32-NEXT:       IsSigned: No
; REL32-NEXT:       FixupBitValue: 0
; REL32-NEXT:       Length: 26
; REL32-NEXT:       Type: R_RBA (0x18)
; REL32-NEXT:     }
; REL32:       Virtual Address: 0x3A
; REL32-NEXT:       Symbol: global_int_nonzero (21)
; REL32-NEXT:       IsSigned: No
; REL32-NEXT:       FixupBitValue: 0
; REL32-NEXT:       Length: 16
; REL32-NEXT:       Type: R_TOC (0x3)
; REL32-NEXT:     }
; REL32:       Virtual Address: 0x40
; REL32-NEXT:       Symbol: .__get_tpointer (1)
; REL32-NEXT:       IsSigned: No
; REL32-NEXT:       FixupBitValue: 0
; REL32-NEXT:       Length: 26
; REL32-NEXT:       Type: R_RBA (0x18)
; REL32-NEXT:     }
; REL32:       Virtual Address: 0x6A
; REL32-NEXT:       Symbol: intern_int_zero (19)
; REL32-NEXT:       IsSigned: No
; REL32-NEXT:       FixupBitValue: 0
; REL32-NEXT:       Length: 16
; REL32-NEXT:       Type: R_TOC (0x3)
; REL32-NEXT:     }
; REL32:       Virtual Address: 0x70
; REL32-NEXT:       Symbol: .__get_tpointer (1)
; REL32-NEXT:       IsSigned: No
; REL32-NEXT:       FixupBitValue: 0
; REL32-NEXT:       Length: 26
; REL32-NEXT:       Type: R_RBA (0x18)
; REL32-NEXT:     }
; REL32:       Virtual Address: 0xAC
; REL32-NEXT:       Symbol: intern_int_zero (27)
; REL32-NEXT:       IsSigned: No
; REL32-NEXT:       FixupBitValue: 0
; REL32-NEXT:       Length: 32
; REL32-NEXT:       Type: R_TLS_IE (0x21)
; REL32-NEXT:     }
; REL32:       Virtual Address: 0xB0
; REL32-NEXT:       Symbol: global_int_nonzero (25)
; REL32-NEXT:       IsSigned: No
; REL32-NEXT:       FixupBitValue: 0
; REL32-NEXT:       Length: 32
; REL32-NEXT:       Type: R_TLS_IE (0x21)
; REL32-NEXT:     }

; SYM32:      File: {{.*}}aix-tls-ie-xcoff-reloc.ll.tmp.o
; SYM32-NEXT: Format: aixcoff-rs6000
; SYM32-NEXT: Arch: powerpc
; SYM32-NEXT: AddressSize: 32bit
; SYM32-NEXT: Symbols [
; SYM32:     Index: 19
; SYM32-NEXT:     Name: intern_int_zero
; SYM32-NEXT:     Value (RelocatableAddress): 0xAC
; SYM32-NEXT:     Section: .data
; SYM32-NEXT:     Type: 0x0
; SYM32-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM32-NEXT:     NumberOfAuxEntries: 1
; SYM32-NEXT:     CSECT Auxiliary Entry {
; SYM32-NEXT:       Index: 20
; SYM32-NEXT:       SectionLen: 4
; SYM32-NEXT:       ParameterHashIndex: 0x0
; SYM32-NEXT:       TypeChkSectNum: 0x0
; SYM32-NEXT:       SymbolAlignmentLog2: 2
; SYM32-NEXT:       SymbolType: XTY_SD (0x1)
; SYM32-NEXT:       StorageMappingClass: XMC_TC (0x3)
; SYM32-NEXT:       StabInfoIndex: 0x0
; SYM32-NEXT:       StabSectNum: 0x0
; SYM32-NEXT:     }
; SYM32-NEXT:   }
; SYM32:     Index: 21
; SYM32-NEXT:     Name: global_int_nonzero
; SYM32-NEXT:     Value (RelocatableAddress): 0xB0
; SYM32-NEXT:     Section: .data
; SYM32-NEXT:     Type: 0x0
; SYM32-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM32-NEXT:     NumberOfAuxEntries: 1
; SYM32-NEXT:     CSECT Auxiliary Entry {
; SYM32-NEXT:       Index: 22
; SYM32-NEXT:       SectionLen: 4
; SYM32-NEXT:       ParameterHashIndex: 0x0
; SYM32-NEXT:       TypeChkSectNum: 0x0
; SYM32-NEXT:       SymbolAlignmentLog2: 2
; SYM32-NEXT:       SymbolType: XTY_SD (0x1)
; SYM32-NEXT:       StorageMappingClass: XMC_TC (0x3)
; SYM32-NEXT:       StabInfoIndex: 0x0
; SYM32-NEXT:       StabSectNum: 0x0
; SYM32-NEXT:     }
; SYM32-NEXT:   }
; SYM32:     Index: 25
; SYM32-NEXT:     Name: global_int_nonzero
; SYM32-NEXT:     Value (RelocatableAddress): 0x0
; SYM32-NEXT:     Section: .tdata
; SYM32-NEXT:     Type: 0x0
; SYM32-NEXT:     StorageClass: C_EXT (0x2)
; SYM32-NEXT:     NumberOfAuxEntries: 1
; SYM32-NEXT:     CSECT Auxiliary Entry {
; SYM32-NEXT:       Index: 26
; SYM32-NEXT:       ContainingCsectSymbolIndex: 23
; SYM32-NEXT:       ParameterHashIndex: 0x0
; SYM32-NEXT:       TypeChkSectNum: 0x0
; SYM32-NEXT:       SymbolAlignmentLog2: 0
; SYM32-NEXT:       SymbolType: XTY_LD (0x2)
; SYM32-NEXT:       StorageMappingClass: XMC_TL (0x14)
; SYM32-NEXT:       StabInfoIndex: 0x0
; SYM32-NEXT:       StabSectNum: 0x0
; SYM32-NEXT:     }
; SYM32-NEXT:   }
; SYM32:     Index: 27
; SYM32-NEXT:     Name: intern_int_zero
; SYM32-NEXT:     Value (RelocatableAddress): 0x4
; SYM32-NEXT:     Section: .tbss
; SYM32-NEXT:     Type: 0x0
; SYM32-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM32-NEXT:     NumberOfAuxEntries: 1
; SYM32-NEXT:     CSECT Auxiliary Entry {
; SYM32-NEXT:       Index: 28
; SYM32-NEXT:       SectionLen: 4
; SYM32-NEXT:       ParameterHashIndex: 0x0
; SYM32-NEXT:       TypeChkSectNum: 0x0
; SYM32-NEXT:       SymbolAlignmentLog2: 2
; SYM32-NEXT:       SymbolType: XTY_CM (0x3)
; SYM32-NEXT:       StorageMappingClass: XMC_UL (0x15)
; SYM32-NEXT:       StabInfoIndex: 0x0
; SYM32-NEXT:       StabSectNum: 0x0
; SYM32-NEXT:     }
; SYM32-NEXT:   }

; DIS32: {{.*}}aix-tls-ie-xcoff-reloc.ll.tmp.o:	file format aixcoff-rs6000
; DIS32: Disassembly of section .text:
; DIS32: (idx: 5) .store_intern_int_zero:
; DIS32: R_TOC        (idx: 19) intern_int_zero[TC]
; DIS32: R_RBA        (idx: 1) .__get_tpointer[PR]
; DIS32: blr
; DIS32: (idx: 7) .load_global_int_nonzero:
; DIS32: R_TOC        (idx: 21) global_int_nonzero[TC]
; DIS32: R_RBA        (idx: 1) .__get_tpointer[PR]
; DIS32: blr
; DIS32: (idx: 9) .load_intern_int_zero:
; DIS32: R_TOC        (idx: 19) intern_int_zero[TC]
; DIS32: R_RBA        (idx: 1) .__get_tpointer[PR]
; DIS32: blr

; DIS32: Disassembly of section .data:
; DIS32: (idx: 11) store_intern_int_zero[DS]:
; DIS32: R_POS        (idx: 5) .store_intern_int_zero
; DIS32: R_POS        (idx: 17) TOC[TC0]
; DIS32: (idx: 13) load_global_int_nonzero[DS]:
; DIS32: R_POS        (idx: 7) .load_global_int_nonzero
; DIS32: R_POS        (idx: 17) TOC[TC0]
; DIS32: (idx: 15) load_intern_int_zero[DS]:
; DIS32: R_POS        (idx: 9) .load_intern_int_zero
; DIS32: R_POS        (idx: 17) TOC[TC0]
; DIS32: (idx: 19) intern_int_zero[TC]:
; DIS32: R_TLS_IE     (idx: 27) intern_int_zero[UL]
; DIS32: (idx: 21) global_int_nonzero[TC]:
; DIS32: R_TLS_IE     (idx: 25) global_int_nonzero

; DIS32: Disassembly of section .tdata:
; DIS32: (idx: 25) global_int_nonzero:

; DIS32: Disassembly of section .tbss:
; DIS32: (idx: 27) intern_int_zero[UL]:
