// This test checks that the SEH directives emit the correct unwind data.

// RUN: llvm-mc -triple aarch64-pc-win32 -filetype=obj %s | llvm-readobj -S -r - | FileCheck %s

// CHECK:      Sections [
// CHECK:        Section {
// CHECK:          Name: .text
// CHECK:          RelocationCount: 0
// CHECK:          Characteristics [
// CHECK-NEXT:       ALIGN_4BYTES
// CHECK-NEXT:       CNT_CODE
// CHECK-NEXT:       MEM_EXECUTE
// CHECK-NEXT:       MEM_READ
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK:        Section {
// CHECK:          Name: .xdata
// CHECK:          RawDataSize: 24
// CHECK:          RelocationCount: 1
// CHECK:          Characteristics [
// CHECK-NEXT:       ALIGN_4BYTES
// CHECK-NEXT:       CNT_INITIALIZED_DATA
// CHECK-NEXT:       MEM_READ
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK:        Section {
// CHECK:          Name: .pdata
// CHECK:          RelocationCount: 6
// CHECK:          Characteristics [
// CHECK-NEXT:       ALIGN_4BYTES
// CHECK-NEXT:       CNT_INITIALIZED_DATA
// CHECK-NEXT:       MEM_READ
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// CHECK-NEXT: Relocations [
// CHECK-NEXT:   Section (4) .xdata {
// CHECK-NEXT:     0x8 IMAGE_REL_ARM64_ADDR32NB __C_specific_handler
// CHECK-NEXT:   }
// CHECK-NEXT:   Section (5) .pdata {
// CHECK-NEXT:     0x0 IMAGE_REL_ARM64_ADDR32NB func
// CHECK-NEXT:     0x4 IMAGE_REL_ARM64_ADDR32NB .xdata
// CHECK-NEXT:     0x8 IMAGE_REL_ARM64_ADDR32NB func
// CHECK-NEXT:     0xC IMAGE_REL_ARM64_ADDR32NB .xdata
// CHECK-NEXT:     0x10 IMAGE_REL_ARM64_ADDR32NB smallFunc
// CHECK-NEXT:     0x14 IMAGE_REL_ARM64_ADDR32NB .xdata
// CHECK-NEXT:   }
// CHECK-NEXT: ]


    .text
    .globl func
    .def func
    .scl 2
    .type 32
    .endef
    .seh_proc func
func:
    sub sp, sp, #24
    .seh_stackalloc 24
    mov x29, sp
    .seh_endprologue
    .seh_handler __C_specific_handler, @except
    .seh_handlerdata
    .long 0
    .text
    .seh_startchained
    .seh_endprologue
    .seh_endchained
    add sp, sp, #24
    ret
    .seh_endproc

// Test emission of small functions.
    .globl smallFunc
    .def smallFunc
    .scl 2
    .type 32
    .endef
    .seh_proc smallFunc
smallFunc:
    ret
    .seh_endproc
