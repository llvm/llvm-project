; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --symbols %t.o | FileCheck --check-prefixes=OBJ,OBJ32 %s
; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -filetype=obj -o %t64.o < %s
; RUN: llvm-readobj --symbols %t64.o | FileCheck --check-prefixes=OBJ,OBJ64 %s

source_filename = "1.f95"

; OBJ: Name: 1.f95
; OBJ: Source Language ID: TB_Fortran (0x1)
; OBJ32: CPU Version ID: TCPU_COM (0x3)
; OBJ64: CPU Version ID: TCPU_PPC64 (0x2)
