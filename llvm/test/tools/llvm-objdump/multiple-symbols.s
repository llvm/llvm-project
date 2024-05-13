// This test checks the behavior of llvm-objdump's --disassemble-symbols and
// --show-all-symbols options, in the presence of multiple symbols defined at
// the same address in an object file.

// The test input file contains an Arm and a Thumb function, each with two
// function-type symbols defined at its entry point. Also, because it's Arm,
// there's a $a mapping symbol defined at the start of the section, and a $t
// mapping symbol at the point where Arm code stops and Thumb code begins.

// By default, llvm-objdump will pick one of the symbols to disassemble at each
// point where any are defined at all. The tie-break sorting criterion is
// alphabetic, so it will be the alphabetically later symbol in each case: of
// the names aaaa and bbbb for the Arm function it picks bbbb, and of cccc and
// dddd for the Thumb function it picks dddd.

// Including an Arm and a Thumb function also re-checks that these changes to
// the display of symbols doesn't affect the recognition of mapping symbols for
// the purpose of switching disassembly mode.

@ REQUIRES: arm-registered-target

@ RUN: llvm-mc -triple armv8a-unknown-linux -filetype=obj %s -o %t.o

// All the run lines below should generate some subset of this
// display, with different parts included:

@ HEAD:          Disassembly of section .text:
@ HEAD-EMPTY:
@ AMAP-NEXT:     00000000 <$a.0>:
@ AAAA-NEXT:     00000000 <aaaa>:
@ BBBB-NEXT:     00000000 <bbbb>:
@ AABB-NEXT:            0: e0800080      add     r0, r0, r0, lsl #1
@ AABB-NEXT:            4: e12fff1e      bx      lr
@ BOTH-EMPTY:    
@ TMAP-NEXT:     00000008 <$t.1>:
@ CCCC-NEXT:     00000008 <cccc>:
@ DDDD-NEXT:     00000008 <dddd>:
@ CCDD-NEXT:            8: eb00 0080     add.w   r0, r0, r0, lsl #2
@ CCDD-NEXT:            c: 4770          bx      lr

// The default disassembly chooses just the alphabetically later symbol of each
// set, namely bbbb and dddd.

@ RUN: llvm-objdump --triple armv8a -d %t.o | FileCheck --check-prefixes=HEAD,BBBB,AABB,BOTH,DDDD,CCDD %s

// With the --show-all-symbols option, all the symbols are shown, including the
// administrative mapping symbols.

@ RUN: llvm-objdump --triple armv8a --show-all-symbols -d %t.o | FileCheck --check-prefixes=HEAD,AMAP,AAAA,BBBB,AABB,BOTH,TMAP,CCCC,DDDD,CCDD %s

// If we use --disassemble-symbols to ask for the disassembly of aaaa or bbbb
// or both, then we expect the second cccc/dddd function not to appear in the
// output at all. Also, we want to see whichever symbol we asked about, or both
// if we asked about both.

@ RUN: llvm-objdump --triple armv8a --disassemble-symbols=aaaa -d %t.o | FileCheck --check-prefixes=HEAD,AAAA,AABB %s
@ RUN: llvm-objdump --triple armv8a --disassemble-symbols=bbbb -d %t.o | FileCheck --check-prefixes=HEAD,BBBB,AABB %s
@ RUN: llvm-objdump --triple armv8a --disassemble-symbols=aaaa,bbbb -d %t.o | FileCheck --check-prefixes=HEAD,AAAA,BBBB,AABB %s

// With _any_ of those three options and also --show-all-symbols, the
// disassembled code is still limited to just the symbol(s) you asked about,
// but all symbols defined at the same address are mentioned, whether you asked
// about them or not.

@ RUN: llvm-objdump --triple armv8a --disassemble-symbols=aaaa --show-all-symbols -d %t.o | FileCheck --check-prefixes=HEAD,AMAP,AAAA,BBBB,AABB %s
@ RUN: llvm-objdump --triple armv8a --disassemble-symbols=bbbb --show-all-symbols -d %t.o | FileCheck --check-prefixes=HEAD,AMAP,AAAA,BBBB,AABB %s
@ RUN: llvm-objdump --triple armv8a --disassemble-symbols=aaaa,bbbb --show-all-symbols -d %t.o | FileCheck --check-prefixes=HEAD,AMAP,AAAA,BBBB,AABB %s

// Similarly for the Thumb function and its symbols. This time we must check
// that the aaaa/bbbb block of code was not disassembled _before_ the output
// we're expecting.

@ RUN: llvm-objdump --triple armv8a --disassemble-symbols=cccc -d %t.o | FileCheck --check-prefixes=HEAD,CCCC,CCDD %s
@ RUN: llvm-objdump --triple armv8a --disassemble-symbols=dddd -d %t.o | FileCheck --check-prefixes=HEAD,DDDD,CCDD %s
@ RUN: llvm-objdump --triple armv8a --disassemble-symbols=cccc,dddd -d %t.o | FileCheck --check-prefixes=HEAD,CCCC,DDDD,CCDD %s

@ RUN: llvm-objdump --triple armv8a --disassemble-symbols=cccc --show-all-symbols -d %t.o | FileCheck --check-prefixes=HEAD,TMAP,CCCC,DDDD,CCDD %s
@ RUN: llvm-objdump --triple armv8a --disassemble-symbols=dddd --show-all-symbols -d %t.o | FileCheck --check-prefixes=HEAD,TMAP,CCCC,DDDD,CCDD %s
@ RUN: llvm-objdump --triple armv8a --disassemble-symbols=cccc,dddd --show-all-symbols -d %t.o | FileCheck --check-prefixes=HEAD,TMAP,CCCC,DDDD,CCDD %s

.text
.globl aaaa
.globl bbbb
.globl cccc
.globl dddd

.arm
aaaa:
bbbb:
        add     r0, r0, r0, lsl #1
        bx      lr

.thumb
cccc:
dddd:
        add.w   r0, r0, r0, lsl #2
        bx      lr
