// RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj %s | llvm-readobj -u - | FileCheck %s

// Tests for WOD pool sharing / deduplication in V3 unwind info.

// CHECK:       UnwindInformation [

.text

// --- Test 1: suffix sharing ---
// Prolog: push rbx, push rdi, sub rsp, 32 (3 WODs)
// Epilog: add rsp, 32, pop rdi (2 WODs — suffix of prolog)
// Pool should contain only the prolog's 3 WODs; epilog reuses bytes 0..1.
suffix_sharing:
    .seh_proc suffix_sharing
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
// CHECK-LABEL:  StartAddress: suffix_sharing
// Payload = 3 (prolog IPs) + 8 (epilog desc: 6+2) + 3 (WOD pool) = 14 bytes, 7 codes
// CHECK:        PayloadWords: 7
// CHECK:        NumberOfOps: 3
// CHECK:        NumberOfEpilogs: 1
// Prolog WODs: ALLOC_SMALL(32), PUSH(RDI), PUSH(RBX) — 3 bytes
// Epilog uses FirstOp=0 (ALLOC_SMALL + PUSH RDI is a prefix of the pool)
// CHECK:        Epilog [0] {
// CHECK:          NumberOfOps: 2
// CHECK:          FirstOp: 0x0

// --- Test 2: two epilogs sharing pool ---
// Prolog: push rbx, sub rsp, 32
// Epilog 0: add rsp, 32 (1 WOD, distinct from prolog — no push)
// Epilog 1: add rsp, 32 (1 WOD, same as epilog 0)
// Epilog 1 should share epilog 0's FirstOp (not duplicate in pool).
cross_epilog_sharing:
    .seh_proc cross_epilog_sharing
    .seh_unwindversion 3
    .seh_pushreg %rbx
    pushq   %rbx
    .seh_stackalloc 32
    subq    $32, %rsp
    .seh_endprologue
    callq   c
    testl   %eax, %eax
    jle     .L_CROSS_ELSE
    // Epilog 0: only dealloc (no pop rbx — tail call)
    .seh_startepilogue
    .seh_stackalloc 32
    addq    $32, %rsp
    .seh_endepilogue
    jmp     c
.L_CROSS_ELSE:
    // Epilog 1: same as epilog 0
    .seh_startepilogue
    .seh_stackalloc 32
    addq    $32, %rsp
    .seh_endepilogue
    jmp     b
    .seh_endproc
// CHECK-LABEL:  StartAddress: cross_epilog_sharing
// Payload = 2 (prolog IPs) + 10 (epilog descs: 7+3) + 2 (WOD pool) = 14 bytes, 7 codes
// CHECK:        PayloadWords: 7
// CHECK:        NumberOfOps: 2
// CHECK:        NumberOfEpilogs: 2
// Epilog 0: ALLOC_SMALL(32) — found in prolog pool at offset 0
// CHECK:        Epilog [0] {
// CHECK:          NumberOfOps: 1
// CHECK:          FirstOp: 0x0
// CHECK:          ALLOC_SMALL Size=0x20
// Epilog 1: same WODs and same IP offsets — inherited from epilog 0
// CHECK:        Epilog [1] {
// CHECK:          NumberOfOps: 0
// CHECK:          (inherits

// --- Test 3: no-match append ---
// Prolog: push rbx, sub rsp, 32
// Epilog: add rsp, 64 (different alloc size — no pool match)
// Epilog WOD must be appended to the pool.
no_match_append:
    .seh_proc no_match_append
    .seh_unwindversion 3
    .seh_pushreg %rbx
    pushq   %rbx
    .seh_stackalloc 32
    subq    $32, %rsp
    .seh_endprologue
    callq   a
    nop
    .seh_startepilogue
    .seh_stackalloc 64
    addq    $64, %rsp
    .seh_endepilogue
    jmp     b
    .seh_endproc
// CHECK-LABEL:  StartAddress: no_match_append
// Payload = 2 (prolog IPs) + 7 (epilog desc: 6+1) + 3 (WOD pool: 2+1) = 12 bytes, 6 codes
// CHECK:        PayloadWords: 6
// CHECK:        NumberOfOps: 2
// CHECK:        NumberOfEpilogs: 1
// Prolog pool: ALLOC_SMALL(32)=1 byte, PUSH(RBX)=1 byte => 2 bytes
// Epilog WOD: ALLOC_SMALL(64)=1 byte at pool offset 2 (appended)
// CHECK:        Epilog [0] {
// CHECK:          NumberOfOps: 1
// CHECK:          FirstOp: 0x2
// CHECK:          ALLOC_SMALL Size=0x40

// --- Test 4: epilog mirrors prolog exactly ---
// Prolog: push rdi, sub rsp, 32
// Epilog: add rsp, 32, pop rdi (mirror)
// FirstOp should be 0 (exact match in pool).
exact_mirror:
    .seh_proc exact_mirror
    .seh_unwindversion 3
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
    retq
    .seh_endproc
// CHECK-LABEL:  StartAddress: exact_mirror
// Payload = 2 (prolog IPs) + 8 (epilog desc: 6+2) + 2 (WOD pool) = 12 bytes, 6 codes
// CHECK:        PayloadWords: 6
// CHECK:        NumberOfOps: 2
// CHECK:        NumberOfEpilogs: 1
// CHECK:        Epilog [0] {
// CHECK:          NumberOfOps: 2
// CHECK:          FirstOp: 0x0
// CHECK:          ALLOC_SMALL Size=0x20
// CHECK:          PUSH Reg=RDI
// CHECK:       ]
