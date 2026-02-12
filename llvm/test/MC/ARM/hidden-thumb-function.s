// RUN: llvm-mc --triple=thumbv7-none-eabi -filetype=obj %s | llvm-readelf -s - | FileCheck %s

// Switch to Arm state to define a function, then switch back to Thumb
// state before marking it as .hidden. We expect that the function is
// still listed as Arm in the symbol table (low bit clear).

        .arm
        .type   hidden_arm_func, %function
hidden_arm_func: bx lr

        .thumb
        .hidden hidden_arm_func

// CHECK: 00000000 0 FUNC LOCAL HIDDEN {{[0-9]+}} hidden_arm_func

// Define two function symbols in Thumb state, with the .type
// directive before the label in one case and after it in the other.
// We expect that both are marked as Thumb. (This was the _intended_
// use of the 'mark as Thumb' behavior that was accidentally applying
// to .hidden as well.)

        .balign 4
thumb_symbol_before_type:
        .type thumb_symbol_before_type, %function
        bx lr

        .balign 4
        .type thumb_symbol_after_type, %function
thumb_symbol_after_type:
        bx lr

// CHECK: 00000005 0 FUNC LOCAL DEFAULT {{[0-9]+}} thumb_symbol_before_type
// CHECK: 00000009 0 FUNC LOCAL DEFAULT {{[0-9]+}} thumb_symbol_after_type
