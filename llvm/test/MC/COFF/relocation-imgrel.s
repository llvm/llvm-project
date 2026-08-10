// COFF Image-relative relocations
//
// Test that we produce image-relative relocations (IMAGE_REL_I386_DIR32NB
// and IMAGE_REL_AMD64_ADDR32NB) when accessing foo@imgrel.

// RUN: llvm-mc -filetype=obj -triple i686-pc-win32 %s > %t.w32.obj
// RUN: llvm-readobj -r %t.w32.obj | FileCheck --check-prefix=W32 %s
// RUN: llvm-objdump -s %t.w32.obj | FileCheck --check-prefix=W32OBJ %s
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-win32 %s > %t.w64.obj
// RUN: llvm-readobj -r %t.w64.obj | FileCheck --check-prefix=W64 %s
// RUN: llvm-objdump -s %t.w64.obj | FileCheck --check-prefix=W64OBJ %s

.data
foo:
    .long 1
    .long .Llabel@imgrel
    .rva .Llabel, .Llabel + 16, foo, .Lother - 3

.text
.Llabel:
    mov foo@IMGREL(%ebx, %ecx, 4), %eax
.Lother:
    mov foo@imgrel(%ebx, %ecx, 4), %eax

// W32:      Relocations [
// W32-NEXT:   Section (1) .text {
// W32-NEXT:     0x3 IMAGE_REL_I386_DIR32NB foo
// W32-NEXT:     0xA IMAGE_REL_I386_DIR32NB foo
// W32-NEXT:   }
// W32-NEXT:   Section (2) .data {
// W32-NEXT:     0x4 IMAGE_REL_I386_DIR32NB .Llabel
// W32-NEXT:     0x8 IMAGE_REL_I386_DIR32NB .Llabel
// W32-NEXT:     0xC IMAGE_REL_I386_DIR32NB .Llabel
// W32-NEXT:     0x10 IMAGE_REL_I386_DIR32NB foo
// W32-NEXT:     0x14 IMAGE_REL_I386_DIR32NB .Lother
// W32-NEXT:   }
// W32-NEXT: ]

// W32OBJ:      Contents of section .data:
// W32OBJ-NEXT:  0000 01000000 00000000 00000000 10000000
// W32OBJ-NEXT:  0010 00000000 fdffffff

// W64:      Relocations [
// W64-NEXT:   Section (1) .text {
// W64-NEXT:     0x4 IMAGE_REL_AMD64_ADDR32NB foo
// W64-NEXT:     0xC IMAGE_REL_AMD64_ADDR32NB foo
// W64-NEXT:   }
// W64-NEXT:   Section (2) .data {
// W64-NEXT:     0x4 IMAGE_REL_AMD64_ADDR32NB .text
// W64-NEXT:     0x8 IMAGE_REL_AMD64_ADDR32NB .text
// W64-NEXT:     0xC IMAGE_REL_AMD64_ADDR32NB .text
// W64-NEXT:     0x10 IMAGE_REL_AMD64_ADDR32NB foo
// W64-NEXT:     0x14 IMAGE_REL_AMD64_ADDR32NB .text
// W64-NEXT:   }
// W64-NEXT: ]

// W64OBJ:      Contents of section .data:
// W64OBJ-NEXT:  0000 01000000 00000000 00000000 10000000
// W64OBJ-NEXT:  0010 00000000 05000000
