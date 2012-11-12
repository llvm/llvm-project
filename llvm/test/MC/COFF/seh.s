// This test checks that the SEH directives emit the correct unwind data.
// RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj %s | coff-dump.py | FileCheck %s

// CHECK:      Name                 = .xdata
// CHECK-NEXT: VirtualSize
// CHECK-NEXT: VirtualAddress
// CHECK-NEXT: SizeOfRawData        = 52
// CHECK-NEXT: PointerToRawData
// CHECK-NEXT: PointerToRelocations
// CHECK-NEXT: PointerToLineNumbers
// CHECK-NEXT: NumberOfRelocations  = 4
// CHECK-NEXT: NumberOfLineNumbers  = 0
// CHECK-NEXT: Charateristics
// CHECK-NEXT:   IMAGE_SCN_CNT_INITIALIZED_DATA
// CHECK-NEXT:   IMAGE_SCN_ALIGN_4BYTES
// CHECK-NEXT:   IMAGE_SCN_MEM_READ
// CHECK-NEXT: SectionData
// CHECK-NEXT:   09 12 08 03 00 03 0F 30 - 0E 88 00 00 09 64 02 00
// CHECK-NEXT:   04 22 00 1A 00 00 00 00 - 00 00 00 00 21 00 00 00
// CHECK-NEXT:   00 00 00 00 1B 00 00 00 - 00 00 00 00 01 00 00 00
// CHECK-NEXT:   00 00 00 00

    .text
    .globl func
    .def func; .scl 2; .type 32; .endef
    .seh_proc func
func:
    .seh_pushframe @code
    subq $24, %rsp
    .seh_stackalloc 24
    movq %rsi, 16(%rsp)
    .seh_savereg %rsi, 16
    movups %xmm8, (%rsp)
    .seh_savexmm %xmm8, 0
    pushq %rbx
    .seh_pushreg 3
    mov %rsp, %rbx
    .seh_setframe 3, 0
    .seh_endprologue
    .seh_handler __C_specific_handler, @except
    .seh_handlerdata
    .long 0
    .text
    .seh_startchained
    .seh_endprologue
    .seh_endchained
    lea (%rbx), %rsp
    pop %rbx
    addq $24, %rsp
    ret
    .seh_endproc

// Test emission of small functions.
    .globl smallFunc
    .def smallFunc; .scl 2; .type 32; .endef
    .seh_proc smallFunc
smallFunc:
    ret
    .seh_endproc
