; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --symbols %t.o | FileCheck --check-prefixes=OBJ %s
; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -filetype=obj -o %t64.o < %s
; RUN: llvm-readobj --symbols %t64.o | FileCheck --check-prefixes=OBJ %s

source_filename = "1.cpp"

; OBJ: Name: .file
; OBJ: Source Language ID: TB_CPLUSPLUS (0x9)
; OBJ: CPU Version ID: TCPU_ANY (0x5)
; OBJ: Name: 1.cpp
