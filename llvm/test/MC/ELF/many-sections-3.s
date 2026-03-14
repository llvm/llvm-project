// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t
// RUN: llvm-readobj --symbols %t | FileCheck --check-prefix=SYMBOLS %s
// RUN: llvm-nm %t | FileCheck --check-prefix=NM %s

// Test that symbol a has a section that could be confused with common (0xFFF2)
// SYMBOLS:         Name: a
// SYMBOLS-NEXT:    Value: 0x0
// SYMBOLS-NEXT:    Size: 0
// SYMBOLS-NEXT:    Binding: Local (0x0)
// SYMBOLS-NEXT:    Type: None (0x0)
// SYMBOLS-NEXT:    Other: 0
// SYMBOLS-NEXT:    Section: bar (0xFFF2)
// SYMBOLS-NEXT:  }

// Test that we don't get confused
// NM: 0000000000000000 r a

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

gen_sections32768 a
gen_sections16384 b
gen_sections8192 c
gen_sections4096 d
gen_sections2048 e
gen_sections1024 f
gen_sections512 g
gen_sections256 h
gen_sections128 i
gen_sections64 j
gen_sections32 k
gen_sections8 l
gen_sections4 m

        .section foo
        .section foo2
        .section foo3
        .section bar, "a"

a:
