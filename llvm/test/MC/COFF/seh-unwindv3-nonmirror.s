// RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj %s | llvm-readobj -u - | FileCheck %s

// Tests for non-mirror epilog support in V3 unwind info emission.
// These test cases exercise epilogs that differ from the prolog in
// operation count, operation type, or both.

// CHECK:       UnwindInformation [

.text

// --- Test 1: partial restore (epilog has fewer ops than prolog) ---
// Prolog: push rbx, push rdi, sub rsp, 32
// Epilog: add rsp, 32, pop rdi (no pop rbx — tail call path)
partial_restore:
    .seh_proc partial_restore
    .seh_unwindversion 3
    .seh_pushreg %rbx
    pushq   %rbx
    .seh_pushreg %rdi
    pushq   %rdi
    .seh_stackalloc 32
    subq    $32, %rsp
    .seh_endprologue
    callq   a
    nop
    .seh_startepilogue
    .seh_stackalloc 32
    addq    $32, %rsp
    .seh_pushreg %rdi
    popq    %rdi
    .seh_endepilogue
    jmp     b
    .seh_endproc
// CHECK-LABEL:  StartAddress: partial_restore
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog:
// CHECK-NEXT:     PayloadWords:
// CHECK-NEXT:     NumberOfOps: 3
// CHECK-NEXT:     NumberOfEpilogs: 1
// CHECK:          Prolog [3 ops]:
// CHECK:            ALLOC_SMALL Size=0x20
// CHECK:            PUSH Reg=RDI
// CHECK:            PUSH Reg=RBX
// Epilog has 2 ops — partial restore (suffix of prolog).
// FirstOp should point into the prolog's WODs (offset 0 = ALLOC_SMALL).
// CHECK:          Epilog [0] {
// CHECK:            NumberOfOps: 2
// CHECK:            FirstOp: 0x0
// CHECK:            ALLOC_SMALL Size=0x20
// CHECK:            PUSH Reg=RDI

// --- Test 2: different operation type in epilog ---
// Prolog: push rax (padding), sub rsp, 32
// Epilog: add rsp, 32, add rsp, 8 (stackalloc instead of pop for padding)
different_op_type:
    .seh_proc different_op_type
    .seh_unwindversion 3
    .seh_pushreg %rax
    pushq   %rax
    .seh_stackalloc 32
    subq    $32, %rsp
    .seh_endprologue
    callq   a
    nop
    .seh_startepilogue
    .seh_stackalloc 32
    addq    $32, %rsp
    .seh_stackalloc 8
    addq    $8, %rsp
    .seh_endepilogue
    retq
    .seh_endproc
// CHECK-LABEL:  StartAddress: different_op_type
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog:
// CHECK-NEXT:     PayloadWords:
// CHECK-NEXT:     NumberOfOps: 2
// CHECK-NEXT:     NumberOfEpilogs: 1
// CHECK:          Prolog [2 ops]:
// CHECK:            ALLOC_SMALL Size=0x20
// CHECK:            PUSH Reg=RAX
// Epilog has alloc+alloc instead of alloc+push — FirstOp should be past
// the prolog WODs (appended to pool).
// CHECK:          Epilog [0] {
// CHECK:            NumberOfOps: 2
// CHECK-NOT:       FirstOp: 0x0
// CHECK:            ALLOC_SMALL Size=0x20
// CHECK:            ALLOC_SMALL Size=0x8

// --- Test 3: mixed mirror + non-mirror epilogs ---
// Prolog: push rbx, push rdi, sub rsp, 32
// Epilog 0: mirror (add rsp, 32; pop rdi; pop rbx) -> FirstOp=0
// Epilog 1: partial (add rsp, 32; pop rdi) -> FirstOp=0 (suffix)
mixed_epilogs:
    .seh_proc mixed_epilogs
    .seh_unwindversion 3
    .seh_pushreg %rbx
    pushq   %rbx
    .seh_pushreg %rdi
    pushq   %rdi
    .seh_stackalloc 32
    subq    $32, %rsp
    .seh_endprologue
    callq   c
    testl   %eax, %eax
    jle     .L_MIXED_ELSE
    // Mirror epilog (full restore)
    .seh_startepilogue
    .seh_stackalloc 32
    addq    $32, %rsp
    .seh_pushreg %rdi
    popq    %rdi
    .seh_pushreg %rbx
    popq    %rbx
    .seh_endepilogue
    retq
.L_MIXED_ELSE:
    // Partial epilog (no pop rbx — tail call)
    .seh_startepilogue
    .seh_stackalloc 32
    addq    $32, %rsp
    .seh_pushreg %rdi
    popq    %rdi
    .seh_endepilogue
    jmp     b
    .seh_endproc
// CHECK-LABEL:  StartAddress: mixed_epilogs
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog:
// CHECK-NEXT:     PayloadWords:
// CHECK-NEXT:     NumberOfOps: 3
// CHECK-NEXT:     NumberOfEpilogs: 2
// CHECK:          Prolog [3 ops]:
// CHECK:            ALLOC_SMALL Size=0x20
// CHECK:            PUSH Reg=RDI
// CHECK:            PUSH Reg=RBX
// Mirror epilog (FirstOp=0, NumberOfOps=3)
// CHECK:          Epilog [0] {
// CHECK:            NumberOfOps: 3
// CHECK:            FirstOp: 0x0
// CHECK:            ALLOC_SMALL Size=0x20
// CHECK:            PUSH Reg=RDI
// CHECK:            PUSH Reg=RBX
// Partial epilog — different NumberOfOps, so NOT inherited.
// CHECK:          Epilog [1] {
// CHECK:            NumberOfOps: 2
// CHECK:            FirstOp: 0x0
// CHECK:            ALLOC_SMALL Size=0x20
// CHECK:            PUSH Reg=RDI

// --- Test 4: reordered epilog ---
// Prolog: push rbx, push rdi, sub rsp, 32
// Epilog: add rsp, 32, pop rbx, pop rdi (swapped pop order)
// The epilog WODs are different from the prolog's because the registers
// are in a different order.
reordered_epilog:
    .seh_proc reordered_epilog
    .seh_unwindversion 3
    .seh_pushreg %rbx
    pushq   %rbx
    .seh_pushreg %rdi
    pushq   %rdi
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
    .seh_pushreg %rdi
    popq    %rdi
    .seh_endepilogue
    retq
    .seh_endproc
// CHECK-LABEL:  StartAddress: reordered_epilog
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog:
// CHECK-NEXT:     PayloadWords:
// CHECK-NEXT:     NumberOfOps: 3
// CHECK-NEXT:     NumberOfEpilogs: 1
// CHECK:          Prolog [3 ops]:
// CHECK:            ALLOC_SMALL Size=0x20
// CHECK:            PUSH Reg=RDI
// CHECK:            PUSH Reg=RBX
// Epilog has same ops but different register order — distinct WODs.
// CHECK:          Epilog [0] {
// CHECK:            NumberOfOps: 3
// CHECK:            ALLOC_SMALL Size=0x20
// CHECK:            PUSH Reg=RBX
// CHECK:            PUSH Reg=RDI

// --- Test 5: setframe omitted in epilog ---
// Prolog: push rbx, push rdi, push rbp, sub rsp, 48, setframe rbp, 48
// Epilog: add rsp, 48, pop rbp, pop rdi, pop rbx (no setframe)
// This is the pattern that funclets produce.
setframe_omitted:
    .seh_proc setframe_omitted
    .seh_unwindversion 3
    .seh_pushreg %rbp
    pushq   %rbp
    .seh_pushreg %rsi
    pushq   %rsi
    .seh_pushreg %rdi
    pushq   %rdi
    .seh_stackalloc 48
    subq    $48, %rsp
    .seh_setframe %rbp, 48
    leaq    48(%rsp), %rbp
    .seh_endprologue
    callq   a
    nop
    // Epilog omits setframe — ADD RSP subsumes it
    .seh_startepilogue
    .seh_stackalloc 48
    addq    $48, %rsp
    .seh_pushreg %rdi
    popq    %rdi
    .seh_pushreg %rsi
    popq    %rsi
    .seh_pushreg %rbp
    popq    %rbp
    .seh_endepilogue
    retq
    .seh_endproc
// CHECK-LABEL:  StartAddress: setframe_omitted
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog:
// CHECK-NEXT:     PayloadWords:
// CHECK-NEXT:     NumberOfOps: 5
// CHECK-NEXT:     NumberOfEpilogs: 1
// CHECK:          Prolog [5 ops]:
// CHECK:            SET_FPREG Reg=RBP, Offset=0x30
// CHECK:            ALLOC_SMALL Size=0x30
// CHECK:            PUSH Reg=RDI
// CHECK:            PUSH Reg=RSI
// CHECK:            PUSH Reg=RBP
// Epilog has 4 ops (no setframe), FirstOp should be 2 (suffix past setframe WOD).
// CHECK:          Epilog [0] {
// CHECK:            NumberOfOps: 4
// CHECK:            FirstOp: 0x2
// CHECK:            ALLOC_SMALL Size=0x30
// CHECK:            PUSH Reg=RDI
// CHECK:            PUSH Reg=RSI
// CHECK:            PUSH Reg=RBP

// --- Test 6: chained child with its own prolog instructions ---
// Main fragment: push rbx, sub rsp, 32
// Chained child has its own prolog (sub rsp, 16) and a mirror epilog.
// The child's NumberOfOps should be >0 and its pool should contain
// the child's own prolog WODs (not the parent's).
chained_with_prolog:
    .seh_proc chained_with_prolog
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
    // Chained child with its own prolog
    .seh_splitchained
    .seh_stackalloc 16
    subq    $16, %rsp
    .seh_endprologue
    callq   a
    nop
    .seh_startepilogue
    .seh_stackalloc 16
    addq    $16, %rsp
    .seh_endepilogue
    retq
    .seh_endproc
// Main fragment
// CHECK-LABEL:  StartAddress: chained_with_prolog
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK:          NumberOfOps: 2
// CHECK:          NumberOfEpilogs: 1
// CHECK:          Prolog [2 ops]:
// CHECK:            ALLOC_SMALL Size=0x20
// CHECK:            PUSH Reg=RBX
// CHECK:          Epilog [0] {
// CHECK:            NumberOfOps: 2
// CHECK:            FirstOp: 0x0
// Chained child — has its own prolog
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x4)
// CHECK-NEXT:       ChainInfo (0x4)
// CHECK-NEXT:     ]
// CHECK:          NumberOfOps: 1
// CHECK-NEXT:     NumberOfEpilogs: 1
// CHECK:          Prolog [1 ops]:
// CHECK:            ALLOC_SMALL Size=0x10
// CHECK:          Epilog [0] {
// CHECK:            NumberOfOps: 1
// CHECK:            FirstOp: 0x0
// CHECK:            ALLOC_SMALL Size=0x10
// CHECK:       ]
