// RUN: llvm-mc -triple x86_64-pc-win32 -mattr=+push2pop2,+egpr -filetype=obj %s | llvm-readobj -u - | FileCheck %s

// CHECK:       UnwindInformation [

.text

// --- Test 0: file-scope .seh_unwindversion 3 applies to subsequent functions ---
// This sets the default so tests 14+ don't need per-function .seh_unwindversion.
// Tests 1-13 use per-function .seh_unwindversion 3 for explicit testing.

// --- Test 1: simple stack alloc, single epilog at end ---
simple_alloc:
    .seh_proc simple_alloc
    .seh_unwindversion 3
    .seh_stackalloc 40
    subq    $40, %rsp
    .seh_endprologue
    callq   a
    nop
    .seh_startepilogue
    .seh_stackalloc 40
    addq    $40, %rsp
    .seh_endepilogue
    retq
    .seh_endproc
// CHECK-LABEL:  StartAddress: simple_alloc
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog:
// CHECK-NEXT:     PayloadWords:
// CHECK-NEXT:     NumberOfOps: 1
// CHECK-NEXT:     NumberOfEpilogs: 1
// CHECK:          Prolog [1 ops]:
// CHECK:          [0] IP +0x{{[0-9A-F]+}}: ALLOC_SMALL Size=0x28
// CHECK:          Epilog [0] {
// CHECK:            IpOffsetOfLastInstruction:
// CHECK:            [0] IP +0x{{[0-9A-F]+}}: ALLOC_SMALL Size=0x28

// --- Test 2: push + alloc, single epilog ---
push_and_alloc:
    .seh_proc push_and_alloc
    .seh_unwindversion 3
    .seh_pushreg %rbx
    pushq   %rbx
    .seh_stackalloc 32
    subq    $32, %rsp
    .seh_endprologue
    callq   a
    nop
    .seh_startepilogue
    .seh_stackalloc 32
    addq    $32, %rsp
    .seh_pushreg %rbx
    popq    %rbx
    .seh_endepilogue
    retq
    .seh_endproc
// CHECK-LABEL:  StartAddress: push_and_alloc
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog:
// CHECK-NEXT:     PayloadWords:
// CHECK-NEXT:     NumberOfOps: 2
// CHECK-NEXT:     NumberOfEpilogs: 1
// CHECK:          Prolog [2 ops]:
// CHECK:          ALLOC_SMALL Size=0x20
// CHECK:          PUSH Reg=RBX
// CHECK:          Epilog [0] {
// CHECK:            ALLOC_SMALL Size=0x20
// CHECK:            PUSH Reg=RBX

// --- Test 3: multiple pushes + alloc + frame register ---
frame_register:
    .seh_proc frame_register
    .seh_unwindversion 3
    .seh_pushreg %rbp
    pushq   %rbp
    .seh_pushreg %rbx
    pushq   %rbx
    .seh_stackalloc 32
    subq    $32, %rsp
    .seh_setframe %rbp, 32
    leaq    32(%rsp), %rbp
    .seh_endprologue
    callq   a
    nop
    .seh_startepilogue
    .seh_setframe %rbp, 32
    leaq    -32(%rbp), %rsp
    .seh_stackalloc 32
    addq    $32, %rsp
    .seh_pushreg %rbx
    popq    %rbx
    .seh_pushreg %rbp
    popq    %rbp
    .seh_endepilogue
    retq
    .seh_endproc
// CHECK-LABEL:  StartAddress: frame_register
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog:
// CHECK-NEXT:     PayloadWords:
// CHECK-NEXT:     NumberOfOps: 4
// CHECK-NEXT:     NumberOfEpilogs: 1
// CHECK:          Prolog [4 ops]:
// CHECK:          SET_FPREG Reg=RBP, Offset=0x20
// CHECK:          ALLOC_SMALL Size=0x20
// CHECK:          PUSH Reg=RBX
// CHECK:          PUSH Reg=RBP
// CHECK:          Epilog [0] {
// CHECK:            SET_FPREG Reg=RBP, Offset=0x20
// CHECK:            ALLOC_SMALL Size=0x20
// CHECK:            PUSH Reg=RBX
// CHECK:            PUSH Reg=RBP

// --- Test 4: multiple epilogs ---
multiple_epilogs:
    .seh_proc multiple_epilogs
    .seh_unwindversion 3
    .seh_stackalloc 40
    subq    $40, %rsp
    .seh_endprologue
    callq   c
    testl   %eax, %eax
    jle     .L_ELSE_1
    movl    %eax, %ecx
    .seh_startepilogue
    .seh_stackalloc 40
    addq    $40, %rsp
    .seh_endepilogue
    jmp     c
.L_ELSE_1:
    nop
    .seh_startepilogue
    .seh_stackalloc 40
    addq    $40, %rsp
    .seh_endepilogue
    jmp     b
    .seh_endproc
// CHECK-LABEL:  StartAddress: multiple_epilogs
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog:
// CHECK-NEXT:     PayloadWords:
// CHECK-NEXT:     NumberOfOps: 1
// CHECK-NEXT:     NumberOfEpilogs: 2
// CHECK:          Prolog [1 ops]:
// CHECK:          ALLOC_SMALL Size=0x28
// CHECK:          Epilog [0] {
// CHECK:            ALLOC_SMALL Size=0x28
// CHECK:          Epilog [1] {
// CHECK:            (inherits

// --- Test 5: large alloc ---
large_alloc:
    .seh_proc large_alloc
    .seh_unwindversion 3
    .seh_pushreg %rbx
    pushq   %rbx
    .seh_stackalloc 4096
    subq    $4096, %rsp
    .seh_endprologue
    callq   a
    nop
    .seh_startepilogue
    .seh_stackalloc 4096
    addq    $4096, %rsp
    .seh_pushreg %rbx
    popq    %rbx
    .seh_endepilogue
    retq
    .seh_endproc
// CHECK-LABEL:  StartAddress: large_alloc
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog:
// CHECK-NEXT:     PayloadWords:
// CHECK-NEXT:     NumberOfOps: 2
// CHECK-NEXT:     NumberOfEpilogs: 1
// CHECK:          Prolog [2 ops]:
// CHECK:          ALLOC_LARGE Size=0x1000
// CHECK:          PUSH Reg=RBX
// CHECK:          Epilog [0] {
// CHECK:            ALLOC_LARGE Size=0x1000
// CHECK:            PUSH Reg=RBX

// --- Test 6: handler ---
with_handler:
    .seh_proc with_handler
    .seh_handler __C_specific_handler, @unwind, @except
    .seh_unwindversion 3
    .seh_pushreg %rbp
    pushq   %rbp
    .seh_stackalloc 32
    subq    $32, %rsp
    .seh_endprologue
.with_handler_callsite:
    callq   a
    nop
.with_handler_finish:
    .seh_startepilogue
    .seh_stackalloc 32
    addq    $32, %rsp
    .seh_pushreg %rbp
    popq    %rbp
    .seh_endepilogue
    retq
.with_handler_handler:
    jmp     .with_handler_finish
    .seh_handlerdata
    .long   1
    .long   .with_handler_callsite@IMGREL
    .long   .with_handler_finish@IMGREL
    .long   1
    .long   .with_handler_handler@IMGREL
    .text
    .seh_endproc
// CHECK-LABEL:  StartAddress: with_handler
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [
// CHECK:            ExceptionHandler
// CHECK:          ]
// CHECK:          NumberOfOps: 2
// CHECK-NEXT:     NumberOfEpilogs: 1
// CHECK:          Handler: __C_specific_handler

// --- Test 7: XMM register save ---
save_xmm:
    .seh_proc save_xmm
    .seh_unwindversion 3
    .seh_pushreg %rbx
    pushq   %rbx
    .seh_stackalloc 48
    subq    $48, %rsp
    .seh_savexmm %xmm6, 32
    movaps  %xmm6, 32(%rsp)
    .seh_endprologue
    callq   a
    nop
    .seh_startepilogue
    .seh_savexmm %xmm6, 32
    movaps  32(%rsp), %xmm6
    .seh_stackalloc 48
    addq    $48, %rsp
    .seh_pushreg %rbx
    popq    %rbx
    .seh_endepilogue
    retq
    .seh_endproc
// CHECK-LABEL:  StartAddress: save_xmm
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog:
// CHECK-NEXT:     PayloadWords:
// CHECK-NEXT:     NumberOfOps: 3
// CHECK-NEXT:     NumberOfEpilogs: 1
// CHECK:          Prolog [3 ops]:
// CHECK:          SAVE_XMM128 Reg=XMM6, Disp=0x20
// CHECK:          ALLOC_SMALL Size=0x30
// CHECK:          PUSH Reg=RBX
// CHECK:          Epilog [0] {
// CHECK:            SAVE_XMM128 Reg=XMM6, Disp=0x20
// CHECK:            ALLOC_SMALL Size=0x30
// CHECK:            PUSH Reg=RBX

// --- Test 8: non-volatile register save (mov to stack) ---
save_nonvol:
    .seh_proc save_nonvol
    .seh_unwindversion 3
    .seh_stackalloc 48
    subq    $48, %rsp
    .seh_savereg %rbx, 40
    movq    %rbx, 40(%rsp)
    .seh_savereg %rsi, 32
    movq    %rsi, 32(%rsp)
    .seh_endprologue
    callq   a
    nop
    .seh_startepilogue
    .seh_savereg %rsi, 32
    movq    32(%rsp), %rsi
    .seh_savereg %rbx, 40
    movq    40(%rsp), %rbx
    .seh_stackalloc 48
    addq    $48, %rsp
    .seh_endepilogue
    retq
    .seh_endproc
// CHECK-LABEL:  StartAddress: save_nonvol
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog:
// CHECK-NEXT:     PayloadWords:
// CHECK-NEXT:     NumberOfOps: 3
// CHECK-NEXT:     NumberOfEpilogs: 1
// CHECK:          Prolog [3 ops]:
// CHECK:          SAVE_NONVOL Reg=RSI, Disp=0x20
// CHECK:          SAVE_NONVOL Reg=RBX, Disp=0x28
// CHECK:          ALLOC_SMALL Size=0x30
// CHECK:          Epilog [0] {
// CHECK:            SAVE_NONVOL Reg=RSI, Disp=0x20
// CHECK:            SAVE_NONVOL Reg=RBX, Disp=0x28
// CHECK:            ALLOC_SMALL Size=0x30

// --- Test 9: pushframe (machine frame) ---
pushframe:
    .seh_proc pushframe
    .seh_unwindversion 3
    .seh_pushframe @code
    .seh_stackalloc 40
    subq    $40, %rsp
    .seh_endprologue
    nop
    retq
    .seh_endproc
// CHECK-LABEL:  StartAddress: pushframe
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog:
// CHECK-NEXT:     PayloadWords:
// CHECK-NEXT:     NumberOfOps: 2
// CHECK-NEXT:     NumberOfEpilogs: 0
// CHECK:          Prolog [2 ops]:
// CHECK:          ALLOC_SMALL Size=0x28
// CHECK:          PUSH_CANONICAL_FRAME Type=1

// --- Test 10: chained unwind info (sub-fragment split) ---
chained:
    .seh_proc chained
    .seh_unwindversion 3
    .seh_stackalloc 40
    subq    $40, %rsp
    .seh_endprologue
    callq   c
    testl   %eax, %eax
    jle     .L_CHAIN_ELSE
    movl    %eax, %ecx
    .seh_startepilogue
    .seh_stackalloc 40
    addq    $40, %rsp
    .seh_endepilogue
    jmp     c
    .seh_splitchained
    .seh_endprologue
.L_CHAIN_ELSE:
    nop
    .seh_startepilogue
    .seh_stackalloc 40
    addq    $40, %rsp
    .seh_endepilogue
    jmp     b
    .seh_endproc
// First fragment: the main function with one epilog.
// CHECK-LABEL:  StartAddress: chained
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog: 0x4
// CHECK-NEXT:     PayloadWords:
// CHECK-NEXT:     NumberOfOps: 1
// CHECK-NEXT:     NumberOfEpilogs: 1
// CHECK:          ALLOC_SMALL Size=0x28
// CHECK:          Epilog [0] {
// CHECK:            ALLOC_SMALL Size=0x28
// Second fragment: chained, with one epilog.
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x4)
// CHECK-NEXT:       ChainInfo (0x4)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog: 0
// CHECK:          NumberOfOps: 0
// CHECK-NEXT:     NumberOfEpilogs: 1
// CHECK:          Epilog [0] {
// CHECK:            ALLOC_SMALL Size=0x28
// CHECK:          Chained {

// --- Test 11: huge alloc (>= 4GB, uses ALLOC_HUGE) ---
huge_alloc:
    .seh_proc huge_alloc
    .seh_unwindversion 3
    .seh_pushreg %rbx
    pushq   %rbx
    .seh_stackalloc 524288
    subq    $524288, %rsp
    .seh_endprologue
    callq   a
    nop
    .seh_startepilogue
    .seh_stackalloc 524288
    addq    $524288, %rsp
    .seh_pushreg %rbx
    popq    %rbx
    .seh_endepilogue
    retq
    .seh_endproc
// CHECK-LABEL:  StartAddress: huge_alloc
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog:
// CHECK-NEXT:     PayloadWords:
// CHECK-NEXT:     NumberOfOps: 2
// CHECK-NEXT:     NumberOfEpilogs: 1
// CHECK:          Prolog [2 ops]:
// CHECK:          ALLOC_HUGE Size=0x80000
// CHECK:          PUSH Reg=RBX
// CHECK:          Epilog [0] {
// CHECK:            ALLOC_HUGE Size=0x80000
// CHECK:            PUSH Reg=RBX

// --- Test 12: handler + chaining combined ---
handler_and_chain:
    .seh_proc handler_and_chain
    .seh_handler __C_specific_handler, @unwind, @except
    .seh_unwindversion 3
    .seh_stackalloc 40
    subq    $40, %rsp
    .seh_endprologue
    callq   c
    testl   %eax, %eax
    jle     .L_HC_ELSE
.handler_chain_callsite:
    callq   a
    nop
.handler_chain_finish:
    movl    %eax, %ecx
    .seh_startepilogue
    .seh_stackalloc 40
    addq    $40, %rsp
    .seh_endepilogue
    jmp     c
    .seh_splitchained
    .seh_endprologue
.L_HC_ELSE:
    nop
    .seh_startepilogue
    .seh_stackalloc 40
    addq    $40, %rsp
    .seh_endepilogue
    jmp     b
.handler_chain_handler:
    jmp     .handler_chain_finish
    .seh_handlerdata
    .long   1
    .long   .handler_chain_callsite@IMGREL
    .long   .handler_chain_finish@IMGREL
    .long   1
    .long   .handler_chain_handler@IMGREL
    .text
    .seh_endproc
// Main fragment has handler.
// CHECK-LABEL:  StartAddress: handler_and_chain
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [
// CHECK:            ExceptionHandler
// CHECK:          ]
// CHECK:          NumberOfOps: 1
// CHECK-NEXT:     NumberOfEpilogs: 1
// CHECK:          Handler: __C_specific_handler
// Chained fragment.
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x4)
// CHECK-NEXT:       ChainInfo (0x4)
// CHECK-NEXT:     ]

// --- Test 13: no epilog (no-return function, empty payload) ---
no_epilog:
    .seh_proc no_epilog
    .seh_unwindversion 3
    .seh_stackalloc 40
    subq    $40, %rsp
    .seh_endprologue
    callq   a
    int3
    .seh_endproc
// CHECK-LABEL:  StartAddress: no_epilog
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog:
// CHECK-NEXT:     PayloadWords:
// CHECK-NEXT:     NumberOfOps: 1
// CHECK-NEXT:     NumberOfEpilogs: 0
// CHECK:          Prolog [1 ops]:
// CHECK:          ALLOC_SMALL Size=0x28

// --- Test 14: file-scope default applies to function without per-function directive ---
.seh_unwindversion 3
file_scope_default:
    .seh_proc file_scope_default
    .seh_stackalloc 40
    subq    $40, %rsp
    .seh_endprologue
    callq   a
    nop
    .seh_startepilogue
    .seh_stackalloc 40
    addq    $40, %rsp
    .seh_endepilogue
    retq
    .seh_endproc
// CHECK-LABEL:  StartAddress: file_scope_default
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog:
// CHECK-NEXT:     PayloadWords:
// CHECK-NEXT:     NumberOfOps: 1
// CHECK-NEXT:     NumberOfEpilogs: 1
// CHECK:          ALLOC_SMALL Size=0x28
// CHECK:          Epilog [0] {
// CHECK:            ALLOC_SMALL Size=0x28

// --- Test 15: three epilogs ? first is full, 2nd and 3rd inherit ---
three_epilogs:
    .seh_proc three_epilogs
    .seh_unwindversion 3
    .seh_pushreg %rbx
    pushq   %rbx
    .seh_stackalloc 32
    subq    $32, %rsp
    .seh_endprologue
    callq   c
    testl   %eax, %eax
    jle     .L_3E_ELSE
    cmpl    $10, %eax
    jge     .L_3E_LARGE
    nop
    .seh_startepilogue
    .seh_stackalloc 32
    addq    $32, %rsp
    .seh_pushreg %rbx
    popq    %rbx
    .seh_endepilogue
    retq
.L_3E_LARGE:
    nop
    .seh_startepilogue
    .seh_stackalloc 32
    addq    $32, %rsp
    .seh_pushreg %rbx
    popq    %rbx
    .seh_endepilogue
    retq
.L_3E_ELSE:
    nop
    .seh_startepilogue
    .seh_stackalloc 32
    addq    $32, %rsp
    .seh_pushreg %rbx
    popq    %rbx
    .seh_endepilogue
    retq
    .seh_endproc
// CHECK-LABEL:  StartAddress: three_epilogs
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog:
// CHECK-NEXT:     PayloadWords:
// CHECK-NEXT:     NumberOfOps: 2
// CHECK-NEXT:     NumberOfEpilogs: 3
// CHECK:          Prolog [2 ops]:
// CHECK:          ALLOC_SMALL Size=0x20
// CHECK:          PUSH Reg=RBX
// CHECK:          Epilog [0] {
// CHECK:            ALLOC_SMALL Size=0x20
// CHECK:            PUSH Reg=RBX
// CHECK:          Epilog [1] {
// CHECK:            (inherits
// CHECK:          Epilog [2] {
// CHECK:            (inherits

// --- Test 16: push2regs with non-consecutive registers -> WOD_PUSH2 ---
push2_nonconsecutive:
    .seh_proc push2_nonconsecutive
    .seh_push2regs %rbx, %rdi
    push2   %rdi, %rbx
    .seh_stackalloc 32
    subq    $32, %rsp
    .seh_endprologue
    callq   a
    nop
    .seh_startepilogue
    .seh_stackalloc 32
    addq    $32, %rsp
    .seh_push2regs %rbx, %rdi
    pop2    %rdi, %rbx
    .seh_endepilogue
    retq
    .seh_endproc
// CHECK-LABEL:  StartAddress: push2_nonconsecutive
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog:
// CHECK-NEXT:     PayloadWords:
// CHECK-NEXT:     NumberOfOps: 2
// CHECK-NEXT:     NumberOfEpilogs: 1
// CHECK:          Prolog [2 ops]:
// CHECK:          [0] IP +0x{{[0-9A-F]+}}: ALLOC_SMALL Size=0x20
// CHECK:          [1] IP +0x{{[0-9A-F]+}}: PUSH2 Reg1=RBX, Reg2=RDI
// CHECK:          Epilog [0] {
// CHECK:            [0] IP +0x{{[0-9A-F]+}}: ALLOC_SMALL Size=0x20
// CHECK:            [1] IP +0x{{[0-9A-F]+}}: PUSH2 Reg1=RBX, Reg2=RDI

// --- Test 17: push2regs with consecutive registers -> WOD_PUSH_CONSECUTIVE_2 ---
push2_consecutive:
    .seh_proc push2_consecutive
    .seh_push2regs %r12, %r13
    push2   %r13, %r12
    .seh_stackalloc 32
    subq    $32, %rsp
    .seh_endprologue
    callq   a
    nop
    .seh_startepilogue
    .seh_stackalloc 32
    addq    $32, %rsp
    .seh_push2regs %r12, %r13
    pop2    %r13, %r12
    .seh_endepilogue
    retq
    .seh_endproc
// CHECK-LABEL:  StartAddress: push2_consecutive
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog:
// CHECK-NEXT:     PayloadWords:
// CHECK-NEXT:     NumberOfOps: 2
// CHECK-NEXT:     NumberOfEpilogs: 1
// CHECK:          Prolog [2 ops]:
// CHECK:          [0] IP +0x{{[0-9A-F]+}}: ALLOC_SMALL Size=0x20
// CHECK:          [1] IP +0x{{[0-9A-F]+}}: PUSH_CONSECUTIVE_2 Reg=R12 (+R13)
// CHECK:          Epilog [0] {
// CHECK:            [0] IP +0x{{[0-9A-F]+}}: ALLOC_SMALL Size=0x20
// CHECK:            [1] IP +0x{{[0-9A-F]+}}: PUSH_CONSECUTIVE_2 Reg=R12 (+R13)

// --- Test 18: EGPR push (register > 15) uses 5-bit encoding ---
egpr_push:
    .seh_proc egpr_push
    .seh_pushreg %r16
    pushq   %r16
    .seh_stackalloc 32
    subq    $32, %rsp
    .seh_endprologue
    callq   a
    nop
    .seh_startepilogue
    .seh_stackalloc 32
    addq    $32, %rsp
    .seh_pushreg %r16
    popq    %r16
    .seh_endepilogue
    retq
    .seh_endproc
// CHECK-LABEL:  StartAddress: egpr_push
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog:
// CHECK-NEXT:     PayloadWords:
// CHECK-NEXT:     NumberOfOps: 2
// CHECK-NEXT:     NumberOfEpilogs: 1
// CHECK:          Prolog [2 ops]:
// CHECK:          [0] IP +0x{{[0-9A-F]+}}: ALLOC_SMALL Size=0x20
// CHECK:          [1] IP +0x{{[0-9A-F]+}}: PUSH Reg=R16
// CHECK:          Epilog [0] {
// CHECK:            [0] IP +0x{{[0-9A-F]+}}: ALLOC_SMALL Size=0x20
// CHECK:            [1] IP +0x{{[0-9A-F]+}}: PUSH Reg=R16

// --- Test 19: EGPR push2regs with non-consecutive EGPRs ---
egpr_push2_nonconsecutive:
    .seh_proc egpr_push2_nonconsecutive
    .seh_push2regs %r16, %r20
    push2   %r20, %r16
    .seh_stackalloc 32
    subq    $32, %rsp
    .seh_endprologue
    callq   a
    nop
    .seh_startepilogue
    .seh_stackalloc 32
    addq    $32, %rsp
    .seh_push2regs %r16, %r20
    pop2    %r20, %r16
    .seh_endepilogue
    retq
    .seh_endproc
// CHECK-LABEL:  StartAddress: egpr_push2_nonconsecutive
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK:          NumberOfOps: 2
// CHECK:          Prolog [2 ops]:
// CHECK:          [0] IP +0x{{[0-9A-F]+}}: ALLOC_SMALL Size=0x20
// CHECK:          [1] IP +0x{{[0-9A-F]+}}: PUSH2 Reg1=R16, Reg2=R20
// CHECK:          Epilog [0] {
// CHECK:            [0] IP +0x{{[0-9A-F]+}}: ALLOC_SMALL Size=0x20
// CHECK:            [1] IP +0x{{[0-9A-F]+}}: PUSH2 Reg1=R16, Reg2=R20

// --- Test 20: EGPR push2regs with consecutive EGPRs -> PUSH_CONSECUTIVE_2 ---
egpr_push2_consecutive:
    .seh_proc egpr_push2_consecutive
    .seh_push2regs %r16, %r17
    push2   %r17, %r16
    .seh_stackalloc 32
    subq    $32, %rsp
    .seh_endprologue
    callq   a
    nop
    .seh_startepilogue
    .seh_stackalloc 32
    addq    $32, %rsp
    .seh_push2regs %r16, %r17
    pop2    %r17, %r16
    .seh_endepilogue
    retq
    .seh_endproc
// CHECK-LABEL:  StartAddress: egpr_push2_consecutive
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK:          NumberOfOps: 2
// CHECK:          Prolog [2 ops]:
// CHECK:          [0] IP +0x{{[0-9A-F]+}}: ALLOC_SMALL Size=0x20
// CHECK:          [1] IP +0x{{[0-9A-F]+}}: PUSH_CONSECUTIVE_2 Reg=R16 (+R17)
// CHECK:          Epilog [0] {
// CHECK:            [0] IP +0x{{[0-9A-F]+}}: ALLOC_SMALL Size=0x20
// CHECK:            [1] IP +0x{{[0-9A-F]+}}: PUSH_CONSECUTIVE_2 Reg=R16 (+R17)
