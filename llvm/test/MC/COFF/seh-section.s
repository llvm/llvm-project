// This test ensures that, if the section containing a function has a suffix
// (e.g. .text$foo), its unwind info section also has a suffix (.xdata$foo).
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-win32 %s | coff-dump.py | FileCheck %s
// XFAIL: *

// CHECK:      Name                 = .xdata$foo
// CHECK-NEXT: VirtualSize
// CHECK-NEXT: VirtualAddress
// CHECK-NEXT: SizeOfRawData        = 8
// CHECK-NEXT: PointerToRawData
// CHECK-NEXT: PointerToRelocations
// CHECK-NEXT: PointerToLineNumbers
// CHECK-NEXT: NumberOfRelocations  = 0
// CHECK-NEXT: NumberOfLineNumbers  = 0
// CHECK-NEXT: Charateristics
// CHECK-NEXT:   IMAGE_SCN_CNT_INITIALIZED_DATA
// CHECK-NEXT:   IMAGE_SCN_ALIGN_4BYTES
// CHECK-NEXT:   IMAGE_SCN_MEM_READ
// CHECK-NEXT:   IMAGE_SCN_MEM_WRITE
// CHECK-NEXT: SectionData
// CHECK-NEXT:   01 05 02 00 05 50 04 02

    .section .text$foo,"x"
    .globl foo
    .def foo; .scl 2; .type 32; .endef
    .seh_proc foo
foo:
    subq $8, %rsp
    .seh_stackalloc 8
    pushq %rbp
    .seh_pushreg %rbp
    .seh_endprologue
    popq %rbp
    addq $8, %rsp
    ret
    .seh_endproc

