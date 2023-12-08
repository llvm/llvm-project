// This test demonstrates that the alphabetical-order tie breaking between
// multiple symbols defined at the same address is based on the raw symbol
// name, not its demangled version.

@ REQUIRES: arm-registered-target

@ RUN: llvm-mc -triple armv8a-unknown-linux -filetype=obj %s -o %t.o

// All the run lines below should generate some subset of this
// display, with different parts included:

@ COMMON:        Disassembly of section .text:
@
@ RAW-B:         00000000 <_Z4bbbbv>:
@ NICE-B:        00000000 <bbbb()>:
@ NO-B-NOT:                bbbb
@ A:             00000000 <aaaa>:
@ COMMON:               0: e0800080      add     r0, r0, r0, lsl #1
@ COMMON:               4: e12fff1e      bx      lr

// The default disassembly chooses just the alphabetically later symbol, which
// is aaaa, because the leading _ on a mangled name sorts before lowercase
// ASCII.

@ RUN: llvm-objdump --triple armv8a -d %t.o | FileCheck --check-prefixes=COMMON,NO-B,A %s

// With the --show-all-symbols option, bbbb is also shown, in its raw form.

@ RUN: llvm-objdump --triple armv8a --show-all-symbols -d %t.o | FileCheck --check-prefixes=COMMON,RAW-B,A %s

// With --demangle as well, bbbb is demangled, but that doesn't change its
// place in the sorting order.

@ RUN: llvm-objdump --triple armv8a --show-all-symbols --demangle -d %t.o | FileCheck --check-prefixes=COMMON,NICE-B,A %s

.text
.globl aaaa
.globl _Z4bbbv
aaaa:
_Z4bbbbv:
        add r0, r0, r0, lsl #1
        bx lr
