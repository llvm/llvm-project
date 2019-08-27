; RUN: llc -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff < %s | FileCheck %s

; RUN: llc -mtriple powerpc-ibm-aix-xcoff -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --section-headers --file-header %t.o | \
; RUN: FileCheck --check-prefix=OBJ %s

; RUN: not llc -mtriple powerpc64-ibm-aix-xcoff -filetype=obj < %s 2>&1 | \
; RUN: FileCheck --check-prefix=OBJ64 %s
; OBJ64: LLVM ERROR: 64-bit XCOFF object files are not supported yet.

@a = internal global i32 0, align 4
@b = internal global i64 0, align 8
@c = internal global i16 0, align 2

; CHECK:      .lcomm a,4,a,2
; CHECK-NEXT: .lcomm b,8,b,3
; CHECK-NEXT: .lcomm c,2,c,1

; OBJ:      File: {{.*}}aix-xcoff-lcomm.ll.tmp.o
; OBJ-NEXT: Format: aixcoff-rs6000
; OBJ-NEXT: Arch: powerpc
; OBJ-NEXT: AddressSize: 32bit
; OBJ-NEXT: FileHeader {
; OBJ-NEXT:   Magic: 0x1DF
; OBJ-NEXT:   NumberOfSections: 1
; OBJ-NEXT:   TimeStamp:
; OBJ-NEXT:   SymbolTableOffset: 0x3C
; OBJ-NEXT:   SymbolTableEntries: 6
; OBJ-NEXT:   OptionalHeaderSize: 0x0
; OBJ-NEXT:   Flags: 0x0
; OBJ-NEXT: }
; OBJ-NEXT: Sections [
; OBJ-NEXT:   Section {
; OBJ-NEXT:     Index: 1
; OBJ-NEXT:     Name: .bss
; OBJ-NEXT:     PhysicalAddress: 0x0
; OBJ-NEXT:     VirtualAddress: 0x0
; OBJ-NEXT:     Size: 0x14
; OBJ-NEXT:     RawDataOffset: 0x0
; OBJ-NEXT:     RelocationPointer: 0x0
; OBJ-NEXT:     LineNumberPointer: 0x0
; OBJ-NEXT:     NumberOfRelocations: 0
; OBJ-NEXT:     NumberOfLineNumbers: 0
; OBJ-NEXT:     Type: STYP_BSS (0x80)
; OBJ-NEXT:   }
; OBJ-NEXT: ]
