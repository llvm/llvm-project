; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr9  < %s | FileCheck --check-prefixes=ASM %s

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr9 -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --symbols %t.o | FileCheck --check-prefixes=OBJ32 %s
; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr9 -filetype=obj -o %t64.o < %s
; RUN: llvm-readobj --symbols %t64.o | FileCheck --check-prefixes=OBJ64 %s

source_filename = "1.c"

; ASM:      .file   "1.c",,"LLVM{{.*}}"
; ASM-NEXT: .csect ..text..[PR],5
; ASM-NEXT: .rename	..text..[PR],""
; ASM-NEXT: .machine   "PWR9"

; OBJ32:      Symbol {
; OBJ32-NEXT:   Index: 0
; OBJ32-NEXT:   Name: .file
; OBJ32-NEXT:   Value (SymbolTableIndex): 0x0
; OBJ32-NEXT:   Section: N_DEBUG
; OBJ32-NEXT:   Source Language ID: TB_C (0x0)
; OBJ32-NEXT:   CPU Version ID: TCPU_PWR9 (0x1A)
; OBJ32-NEXT:   StorageClass: C_FILE (0x67)
; OBJ32-NEXT:   NumberOfAuxEntries: 2
; OBJ32-NEXT:   File Auxiliary Entry {
; OBJ32-NEXT:     Index: 1
; OBJ32-NEXT:     Name: 1.c
; OBJ32-NEXT:     Type: XFT_FN (0x0)
; OBJ32-NEXT:   }
; OBJ32-NEXT:   File Auxiliary Entry {
; OBJ32-NEXT:     Index: 2
; OBJ32-NEXT:     Name: LLVM
; OBJ32-NEXT:     Type: XFT_CV (0x2)
; OBJ32-NEXT:   }
; OBJ32-NEXT: }

; OBJ64:      Symbol {
; OBJ64-NEXT:   Index: 0
; OBJ64-NEXT:   Name: .file
; OBJ64-NEXT:   Value (SymbolTableIndex): 0x0
; OBJ64-NEXT:   Section: N_DEBUG
; OBJ64-NEXT:   Source Language ID: TB_C (0x0)
; OBJ64-NEXT:   CPU Version ID: TCPU_PWR9 (0x1A)
; OBJ64-NEXT:   StorageClass: C_FILE (0x67)
; OBJ64-NEXT:   NumberOfAuxEntries: 2
; OBJ64-NEXT:   File Auxiliary Entry {
; OBJ64-NEXT:     Index: 1
; OBJ64-NEXT:     Name: 1.c
; OBJ64-NEXT:     Type: XFT_FN (0x0)
; OBJ64-NEXT:     Auxiliary Type: AUX_FILE (0xFC)
; OBJ64-NEXT:   }
; OBJ64-NEXT:   File Auxiliary Entry {
; OBJ64-NEXT:     Index: 2
; OBJ64-NEXT:     Name: LLVM
; OBJ64-NEXT:     Type: XFT_CV (0x2)
; OBJ64-NEXT:     Auxiliary Type: AUX_FILE (0xFC)
; OBJ64-NEXT:   }
; OBJ64-NEXT: }
