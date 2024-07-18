// RUN: llvm-mc -triple=aarch64-none-linux-gnu -filetype=obj < %s | llvm-objdump -t - | FileCheck %s

        .text
        add w0, w0, w0

// .wibble should *not* inherit .text's mapping symbol. It's a completely different section.
        .section .wibble
        add w0, w0, w0

// A new text setion should be able to start with a $d
        .section .starts_data,"ax"
        .word 42

// Changing back to .text should not emit a redundant $x
        .text
        add w0, w0, w0

// Non-text sections do not need the initial $d.
        .section .data,"aw"
        .word 42

        .text
        add w0, w0, w0

// Instructions in error scenarios could cause emission of $x and $d.
        .section .rodata,"a"
        .word 42
        add w0, w0, w0
        .word 42

// With all those constraints, we want:
//   + .text to have $x at 0 and no others
//   + .wibble to have $x at 0
//   + .starts_data to have $d at 0


// CHECK:      0000000000000000 l .text        0000000000000000 $x.0
// CHECK-NEXT: 0000000000000000 l .wibble      0000000000000000 $x.1
// CHECK-NEXT: 0000000000000000 l .starts_data 0000000000000000 $d.2
// CHECK-NEXT: 0000000000000004 l .rodata      0000000000000000 $x.3
// CHECK-NEXT: 0000000000000008 l .rodata      0000000000000000 $d.4
// CHECK-NOT: ${{[adtx]}}

