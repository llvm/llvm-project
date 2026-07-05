// RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t
// RUN: llvm-objdump -d %t | FileCheck %s --check-prefix=NODISASM

// The exact rules of skipping the bytes you can find in the code.
// This test checks that we follow these rules and can force
// dissasembly of zero blocks with the -z and --disassemble-zeroes options.

// NODISASM:       0000000000000000 <main>:
// NODISASM-NEXT:   0:  00 00               addb %al, (%rax)
// NODISASM-NEXT:   2:  00 00               addb %al, (%rax)
// NODISASM-NEXT:   4:  00 00               addb %al, (%rax)
// NODISASM-NEXT:   6:  00 90 00 00 00 00   addb %dl, (%rax)
// NODISASM-NEXT:       ...
// NODISASM-NEXT:   20: 90                  nop
// NODISASM-NEXT:       ...
// NODISASM:      0000000000000031 <foo>:
// NODISASM-NEXT:   31: 00 00               addb %al, (%rax)
// NODISASM-NEXT:   33: 00 00               addb %al, (%rax)
// NODISASM:      0000000000000035 <bar>:
// NODISASM-NEXT:       ...

// Check that with -z we disassemble blocks of zeroes.
// RUN: llvm-objdump -d -z %t | FileCheck %s --check-prefix=DISASM

// DISASM:      0000000000000000 <main>:
// DISASM-NEXT:   0: 00 00              addb %al, (%rax)
// DISASM-NEXT:   2: 00 00              addb %al, (%rax)
// DISASM-NEXT:   4: 00 00              addb %al, (%rax)
// DISASM-NEXT:   6: 00 90 00 00 00 00  addb %dl, (%rax)
// DISASM-NEXT:   c: 00 00              addb %al, (%rax)
// DISASM-NEXT:   e: 00 00              addb %al, (%rax)
// DISASM-NEXT:  10: 00 00              addb %al, (%rax)
// DISASM-NEXT:  12: 00 00              addb %al, (%rax)
// DISASM-NEXT:  14: 00 00              addb %al, (%rax)
// DISASM-NEXT:  16: 00 00              addb %al, (%rax)
// DISASM-NEXT:  18: 00 00              addb %al, (%rax)
// DISASM-NEXT:  1a: 00 00              addb %al, (%rax)
// DISASM-NEXT:  1c: 00 00              addb %al, (%rax)
// DISASM-NEXT:  1e: 00 00              addb %al, (%rax)
// DISASM-NEXT:  20: 90                 nop
// DISASM-NEXT:  21: 00 00              addb %al, (%rax)
// DISASM-NEXT:  23: 00 00              addb %al, (%rax)
// DISASM-NEXT:  25: 00 00              addb %al, (%rax)
// DISASM-NEXT:  27: 00 00              addb %al, (%rax)
// DISASM-NEXT:  29: 00 00              addb %al, (%rax)
// DISASM-NEXT:  2b: 00 00              addb %al, (%rax)
// DISASM-NEXT:  2d: 00 00              addb %al, (%rax)
// DISASM-NEXT:  2f: 00 00              addb %al, (%rax)
// DISASM:      0000000000000031 <foo>:
// DISASM-NEXT:  31: 00 00              addb %al, (%rax)
// DISASM-NEXT:  33: 00 00              addb %al, (%rax)
// DISASM:      0000000000000035 <bar>:
// DISASM-NEXT:  35: 00 00              addb %al, (%rax)
// DISASM-NEXT:  37: 00 00              addb %al, (%rax)
// DISASM-NEXT:  39: 00 00              addb %al, (%rax)
// DISASM-NEXT:  3b: 00 00              addb %al, (%rax)

// Check that --disassemble-zeroes work as alias for -z.
// RUN: llvm-objdump -d --disassemble-zeroes %t | FileCheck %s --check-prefix=DISASM

.text
.globl main
.type main, @function
main:
 .long 0
 .byte 0
 .byte 0
 .byte 0
 nop
 .quad 0
 .quad 0
 .quad 0
 nop
 .quad 0
 .quad 0
foo:
 .long 0
bar:
 .quad 0
