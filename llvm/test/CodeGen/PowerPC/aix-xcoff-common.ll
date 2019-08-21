; RUN: llc -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s

; RUN: llc -mtriple powerpc-ibm-aix-xcoff -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --section-headers --file-header %t.o | \
; RUN: FileCheck --check-prefix=OBJ %s

; RUN: not llc -mtriple powerpc64-ibm-aix-xcoff -filetype=obj -o %t.o 2>&1 \
; RUN: < %s | FileCheck --check-prefix=XCOFF64 %s

; XCOFF64: LLVM ERROR: 64-bit XCOFF object files are not supported yet.

@a = common global i32 0, align 4
@b = common global i64 0, align 8
@c = common global i16 0, align 2

@d = common local_unnamed_addr global double 0.000000e+00, align 8
@f = common local_unnamed_addr global float 0.000000e+00, align 4

@over_aligned = common local_unnamed_addr global double 0.000000e+00, align 32

@array = common local_unnamed_addr global [33 x i8] zeroinitializer, align 1

; CHECK:      .csect .text[PR]
; CHECK-NEXT:  .file
; CHECK-NEXT: .comm   a,4,2
; CHECK-NEXT: .comm   b,8,3
; CHECK-NEXT: .comm   c,2,1
; CHECK-NEXT: .comm   d,8,3
; CHECK-NEXT: .comm   f,4,2
; CHECK-NEXT: .comm   over_aligned,8,5
; CHECK-NEXT: .comm   array,33,0

; OBJ:      File: {{.*}}aix-xcoff-common.ll.tmp.o
; OBJ-NEXT: Format: aixcoff-rs6000
; OBJ-NEXT: Arch: powerpc
; OBJ-NEXT: AddressSize: 32bit
; OBJ-NEXT: FileHeader {
; OBJ-NEXT:   Magic: 0x1DF
; OBJ-NEXT:   NumberOfSections: 1
; OBJ-NEXT:   TimeStamp:
; OBJ-NEXT:   SymbolTableOffset: 0x3C
; OBJ-NEXT:   SymbolTableEntries: 14
; OBJ-NEXT:   OptionalHeaderSize: 0x0
; OBJ-NEXT:   Flags: 0x0
; OBJ-NEXT: }
; OBJ-NEXT: Sections [
; OBJ-NEXT:   Section {
; OBJ-NEXT:     Index: 1
; OBJ-NEXT:     Name: .bss
; OBJ-NEXT:     PhysicalAddress: 0x0
; OBJ-NEXT:     VirtualAddress: 0x0
; OBJ-NEXT:     Size: 0x6C
; OBJ-NEXT:     RawDataOffset: 0x0
; OBJ-NEXT:     RelocationPointer: 0x0
; OBJ-NEXT:     LineNumberPointer: 0x0
; OBJ-NEXT:     NumberOfRelocations: 0
; OBJ-NEXT:     NumberOfLineNumbers: 0
; OBJ-NEXT:     Type: STYP_BSS (0x80)
; OBJ-NEXT:   }
; OBJ-NEXT: ]
