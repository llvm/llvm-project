// This test checks error reporting for missing ending/starting of prologues/epilogues

// RUN: not llvm-mc -triple thumbv7-pc-win32 -filetype=obj -o /dev/null %s 2>&1 | FileCheck %s

// CHECK: error: Stray .seh_endepilogue in func1
// CHECK: error: Prologue in func2 not correctly terminated
// CHECK: error: Epilogue in func3 not correctly terminated
// CHECK: error: Epilogue in func4 not correctly terminated

        .text
        .syntax unified

        .seh_proc func1
func1:
        sub sp, #16
        .seh_stackalloc 16
        .seh_endprologue
        nop
        // Missing .seh_startepilogue
        add sp, #16
        .seh_stackalloc 16
        bx lr
        .seh_nop
        .seh_endepilogue
        .seh_endproc

        .seh_proc func2
func2:
        sub sp, #16
        .seh_stackalloc 16
        // Missing .seh_endprologue
        nop
        .seh_startepilogue
        add sp, #16
        .seh_stackalloc 16
        bx lr
        .seh_nop
        .seh_endepilogue
        .seh_endproc

        .seh_proc func3
func3:
        sub sp, #16
        .seh_stackalloc 16
        .seh_endprologue
        nop
        .seh_startepilogue
        add sp, #16
        .seh_stackalloc 16
        bx lr
        .seh_nop
        // Missing .seh_endepilogue
        .seh_endproc

        .seh_proc func4
func4:
        sub sp, #16
        .seh_stackalloc 16
        .seh_endprologue
        nop
        .seh_startepilogue
        add sp, #16
        .seh_stackalloc 16
        bx lr
        .seh_nop
        // Missing .seh_endepilogue
        nop
        .seh_startepilogue
        add sp, #16
        .seh_stackalloc 16
        bx lr
        .seh_nop
        .seh_endepilogue
        .seh_endproc
