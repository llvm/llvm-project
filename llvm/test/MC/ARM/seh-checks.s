// This test checks error reporting for mismatched prolog/epilog lengths

// RUN: not llvm-mc -triple thumbv7-pc-win32 -filetype=obj -o /dev/null %s 2>&1 | FileCheck %s

// CHECK-NOT: func1
// CHECK: error: Incorrect size for func2 epilogue: 6 bytes of instructions in range, but .seh directives corresponding to 4 bytes
// CHECK: error: Incorrect size for func3 prologue: 4 bytes of instructions in range, but .seh directives corresponding to 2 bytes

        .text
        .syntax unified

        .seh_proc func1
func1:
        // Instruction with indeterminate length
        b other
        .seh_endprologue
        nop
        .seh_startepilogue
        // The p2align causes the length of the epilogue to be unknown, so
        // we can't report errors about the mismatch here.
        .p2align 1
        pop {r4-r7-lr}
        .seh_save_regs {r4-r7,lr}
        bx lr
        .seh_nop
        .seh_endepilogue
        .seh_endproc

        .seh_proc func2
func2:
        .seh_endprologue
        nop
        .seh_startepilogue
        // As we're popping into lr instead of directly into pc, this pop
        // becomes a wide instruction.
        pop {r4-r7,lr}
        // The directive things we're making a narrow instruction, which
        // is wrong.
        .seh_save_regs {r4-r7,lr}
        bx lr
        .seh_nop
        .seh_endepilogue
        .seh_endproc

        .seh_proc func3
func3:
        nop.w
        .seh_nop
        .seh_endprologue
        nop
        .seh_endproc
