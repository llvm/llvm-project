// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t
// RUN: llvm-readobj -S %t | FileCheck --check-prefix=SECTIONS %s
// RUN: llvm-readobj --symbols %t | FileCheck --check-prefix=SYMBOLS %s

// Test that we create a .symtab_shndx if a symbol points to a section
// numbered SHN_LORESERVE (0xFF00) or higher.

// SECTIONS: Name: .symtab_shndx

// Test that we don't create a symbol for the symtab_shndx section.
// SYMBOLS-NOT: symtab_shndx


// Test that both a and b show up in the correct section.
// SYMBOLS:         Name: a
// SYMBOLS-NEXT:    Value: 0x0
// SYMBOLS-NEXT:    Size: 0
// SYMBOLS-NEXT:    Binding: Local (0x0)
// SYMBOLS-NEXT:    Type: None (0x0)
// SYMBOLS-NEXT:    Other: 0
// SYMBOLS-NEXT:    Section: dm (0xFF00)
// SYMBOLS-NEXT:  }
// SYMBOLS-NEXT:  Symbol {
// SYMBOLS-NEXT:    Name: b
// SYMBOLS-NEXT:    Value: 0x1
// SYMBOLS-NEXT:    Size: 0
// SYMBOLS-NEXT:    Binding: Local (0x0)
// SYMBOLS-NEXT:    Type: None (0x0)
// SYMBOLS-NEXT:    Other: 0
// SYMBOLS-NEXT:    Section: dm (0xFF00)
// SYMBOLS-NEXT:  }


// Test that this file has one section too many.
// SYMBOLS:         Name: dm (0)
// SYMBOLS:         Value: 0x0
// SYMBOLS:         Size: 0
// SYMBOLS:         Binding: Local (0x0)
// SYMBOLS:         Type: Section (0x3)
// SYMBOLS:         Other: 0
// SYMBOLS:         Section: dm (0xFF00)
// SYMBOLS-NEXT:  }
// SYMBOLS-NEXT:]

.macro gen_sections4 x
        .section a\x
        .section b\x
        .section c\x
        .section d\x
.endm

.macro gen_sections8 x
        gen_sections4 a\x
        gen_sections4 b\x
.endm

.macro gen_sections16 x
        gen_sections8 a\x
        gen_sections8 b\x
.endm

.macro gen_sections32 x
        gen_sections16 a\x
        gen_sections16 b\x
.endm

.macro gen_sections64 x
        gen_sections32 a\x
        gen_sections32 b\x
.endm

.macro gen_sections128 x
        gen_sections64 a\x
        gen_sections64 b\x
.endm

.macro gen_sections256 x
        gen_sections128 a\x
        gen_sections128 b\x
.endm

.macro gen_sections512 x
        gen_sections256 a\x
        gen_sections256 b\x
.endm

.macro gen_sections1024 x
        gen_sections512 a\x
        gen_sections512 b\x
.endm

.macro gen_sections2048 x
        gen_sections1024 a\x
        gen_sections1024 b\x
.endm

.macro gen_sections4096 x
        gen_sections2048 a\x
        gen_sections2048 b\x
.endm

.macro gen_sections8192 x
        gen_sections4096 a\x
        gen_sections4096 b\x
.endm

.macro gen_sections16384 x
        gen_sections8192 a\x
        gen_sections8192 b\x
.endm

.macro gen_sections32768 x
        gen_sections16384 a\x
        gen_sections16384 b\x
.endm

        .section foo
        .section bar

gen_sections32768 a
gen_sections16384 b
gen_sections8192 c
gen_sections4096 d
gen_sections2048 e
gen_sections1024 f
gen_sections512 g
gen_sections128 h
gen_sections64 i
gen_sections32 j
gen_sections16 k
gen_sections8 l
gen_sections4 m

a:
b = a + 1
.long dm
