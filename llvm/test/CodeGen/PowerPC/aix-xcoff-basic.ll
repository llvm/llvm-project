; RUN: llc -mtriple powerpc-ibm-aix-xcoff -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --file-headers %t.o | FileCheck %s

; RUN: not llc -mtriple powerpc64-ibm-aix-xcoff -filetype=obj < %s 2>&1 | \
; RUN: FileCheck --check-prefix=64BIT %s

; RUN: llc -mtriple powerpc-ibm-aix-xcoff < %s | \
; RUN: FileCheck --check-prefix=ASM %s

; RUN: llc -mtriple powerpc64-ibm-aix-xcoff < %s | \
; RUN: FileCheck --check-prefix=ASM %s

target datalayout = "E-m:e-p:32:32-i64:64-n32"
target triple = "powerpc-unknown-aix"

; NOTE: The object file output and the assembly file output do not describe the
;       same abstract XCOFF content due to the limited amount of functionality
;       implemented.

; CHECK:      Format: aixcoff-rs6000
; CHECK-NEXT: Arch: powerpc
; CHECK-NEXT: AddressSize: 32bit
; CHECK-NEXT: FileHeader {
; CHECK-NEXT:   Magic: 0x1DF
; CHECK-NEXT:   NumberOfSections: 0
; CHECK-NEXT:   TimeStamp: None (0x0)
; CHECK-NEXT:   SymbolTableOffset: 0x0
; CHECK-NEXT:   SymbolTableEntries: 0
; CHECK-NEXT:   OptionalHeaderSize: 0x0
; CHECK-NEXT:   Flags: 0x0
; CHECK-NEXT: }

; 64BIT: LLVM ERROR: 64-bit XCOFF object files are not supported yet.

; The csect does not need to be present, but LLVM's default behavior when
; emitting asm is to start the file with the .text section.
; ASM: .csect .text[PR]
