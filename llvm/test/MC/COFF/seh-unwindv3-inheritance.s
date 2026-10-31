// RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj %s | llvm-readobj -u - | FileCheck %s

// Tests for epilog descriptor inheritance in V3 unwind info.

// CHECK:       UnwindInformation [

.text

// --- Test 1: two identical mirror epilogs -> second inherits ---
// Both epilogs have the same WODs, same FirstOp, same IP offsets.
same_ops_inherit:
    .seh_proc same_ops_inherit
    .seh_unwindversion 3
    .seh_pushreg %rbx
    pushq   %rbx
    .seh_stackalloc 32
    subq    $32, %rsp
    .seh_endprologue
    callq   c
    testl   %eax, %eax
    jle     .L_INHERIT_ELSE
    // Epilog 0: mirror
    .seh_startepilogue
    .seh_stackalloc 32
    addq    $32, %rsp
    .seh_pushreg %rbx
    popq    %rbx
    .seh_endepilogue
    retq
.L_INHERIT_ELSE:
    // Epilog 1: identical mirror
    .seh_startepilogue
    .seh_stackalloc 32
    addq    $32, %rsp
    .seh_pushreg %rbx
    popq    %rbx
    .seh_endepilogue
    retq
    .seh_endproc
// CHECK-LABEL:  StartAddress: same_ops_inherit
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog: 0x5
// CHECK-NEXT:     PayloadWords: 8
// CHECK-NEXT:     NumberOfOps: 2
// CHECK-NEXT:     NumberOfEpilogs: 2
// CHECK-NEXT:     Prolog [2 ops]:
// CHECK-NEXT:       [0] IP +0x0001: ALLOC_SMALL Size=0x20
// CHECK-NEXT:       [1] IP +0x0000: PUSH Reg=RBX
// CHECK-NEXT:     Epilog [0] {
// CHECK-NEXT:       Flags [ (0x0)
// CHECK-NEXT:       ]
// CHECK-NEXT:       EpilogOffset: -0x6
// CHECK-NEXT:       NumberOfOps: 2
// CHECK-NEXT:       FirstOp: 0x0
// CHECK-NEXT:       IpOffsetOfLastInstruction: 0x5
// CHECK-NEXT:       [0] IP +0x0000: ALLOC_SMALL Size=0x20
// CHECK-NEXT:       [1] IP +0x0004: PUSH Reg=RBX
// CHECK-NEXT:     }
// Epilog 1 inherits (same FirstOp, same NumberOfOps, same IP offsets).
// CHECK-NEXT:     Epilog [1] {
// CHECK-NEXT:       Flags [ (0x0)
// CHECK-NEXT:       ]
// CHECK-NEXT:       EpilogOffset: -0xC
// CHECK-NEXT:       NumberOfOps: 0
// CHECK-NEXT:       (inherits from base epilog: FirstOp=0x0, IpOffsetOfLastInstruction=0x5, 2 ops)
// CHECK-NEXT:     }

// --- Test 2: two epilogs, different NumberOfOps -> no inheritance ---
// Epilog 0: mirror (2 ops), Epilog 1: partial (1 op)
different_numops:
    .seh_proc different_numops
    .seh_unwindversion 3
    .seh_pushreg %rbx
    pushq   %rbx
    .seh_stackalloc 32
    subq    $32, %rsp
    .seh_endprologue
    callq   c
    testl   %eax, %eax
    jle     .L_DIFFOPS_ELSE
    // Epilog 0: full mirror
    .seh_startepilogue
    .seh_stackalloc 32
    addq    $32, %rsp
    .seh_pushreg %rbx
    popq    %rbx
    .seh_endepilogue
    retq
.L_DIFFOPS_ELSE:
    // Epilog 1: partial -> only dealloc, no pop
    .seh_startepilogue
    .seh_stackalloc 32
    addq    $32, %rsp
    .seh_endepilogue
    jmp     b
    .seh_endproc
// CHECK-LABEL:  StartAddress: different_numops
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog: 0x5
// CHECK-NEXT:     PayloadWords: 10
// CHECK-NEXT:     NumberOfOps: 2
// CHECK-NEXT:     NumberOfEpilogs: 2
// CHECK-NEXT:     Prolog [2 ops]:
// CHECK-NEXT:       [0] IP +0x0001: ALLOC_SMALL Size=0x20
// CHECK-NEXT:       [1] IP +0x0000: PUSH Reg=RBX
// V3 emits epilog descriptors in descending address order (tail-relative
// offsets), so the later partial (1-op) epilog is Epilog [0].
// CHECK-NEXT:     Epilog [0] {
// CHECK-NEXT:       Flags [ (0x0)
// CHECK-NEXT:       ]
// CHECK-NEXT:       EpilogOffset: -0x9
// CHECK-NEXT:       NumberOfOps: 1
// CHECK-NEXT:       FirstOp: 0x0
// CHECK-NEXT:       IpOffsetOfLastInstruction: 0x4
// CHECK-NEXT:       [0] IP +0x0000: ALLOC_SMALL Size=0x20
// CHECK-NEXT:     }
// Epilog 1: different NumberOfOps -> gets its own full descriptor, not inherited.
// CHECK-NEXT:     Epilog [1] {
// CHECK-NEXT:       Flags [ (0x0)
// CHECK-NEXT:       ]
// CHECK-NEXT:       EpilogOffset: -0xF
// CHECK-NEXT:       NumberOfOps: 2
// CHECK-NEXT:       FirstOp: 0x0
// CHECK-NEXT:       IpOffsetOfLastInstruction: 0x5
// CHECK-NEXT:       [0] IP +0x0000: ALLOC_SMALL Size=0x20
// CHECK-NEXT:       [1] IP +0x0004: PUSH Reg=RBX
// CHECK-NEXT:     }

// --- Test 3: two epilogs, same NumberOfOps but different WODs -> no inheritance ---
// Epilog 0: pop rdi, Epilog 1: pop rbx (different register in single-op epilog)
different_wods:
    .seh_proc different_wods
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
    jle     .L_DIFFWOD_ELSE
    // Epilog 0: pop rdi only
    .seh_startepilogue
    .seh_pushreg %rdi
    popq    %rdi
    .seh_endepilogue
    jmp     c
.L_DIFFWOD_ELSE:
    // Epilog 1: pop rbx only (different register)
    .seh_startepilogue
    .seh_pushreg %rbx
    popq    %rbx
    .seh_endepilogue
    jmp     b
    .seh_endproc
// CHECK-LABEL:  StartAddress: different_wods
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog: 0x6
// CHECK-NEXT:     PayloadWords: 10
// CHECK-NEXT:     NumberOfOps: 3
// CHECK-NEXT:     NumberOfEpilogs: 2
// CHECK-NEXT:     Prolog [3 ops]:
// CHECK-NEXT:       [0] IP +0x0002: ALLOC_SMALL Size=0x20
// CHECK-NEXT:       [1] IP +0x0001: PUSH Reg=RDI
// CHECK-NEXT:       [2] IP +0x0000: PUSH Reg=RBX
// Descriptors are in descending address order, so the later epilog
// (PUSH RBX) is Epilog [0]; it shares the prolog's RBX WOD (FirstOp=0x2).
// CHECK-NEXT:     Epilog [0] {
// CHECK-NEXT:       Flags [ (0x0)
// CHECK-NEXT:       ]
// CHECK-NEXT:       EpilogOffset: -0x6
// CHECK-NEXT:       NumberOfOps: 1
// CHECK-NEXT:       FirstOp: 0x2
// CHECK-NEXT:       IpOffsetOfLastInstruction: 0x1
// CHECK-NEXT:       [0] IP +0x0000: PUSH Reg=RBX
// CHECK-NEXT:     }
// Epilog 1: PUSH RDI -> different FirstOp (0x1), so NOT inherited.
// CHECK-NEXT:     Epilog [1] {
// CHECK-NEXT:       Flags [ (0x0)
// CHECK-NEXT:       ]
// CHECK-NEXT:       EpilogOffset: -0xC
// CHECK-NEXT:       NumberOfOps: 1
// CHECK-NEXT:       FirstOp: 0x1
// CHECK-NEXT:       IpOffsetOfLastInstruction: 0x1
// CHECK-NEXT:       [0] IP +0x0000: PUSH Reg=RDI
// CHECK-NEXT:     }

// --- Test 4: two identical LARGE mirror epilogs -> second inherits ---
// Each epilog has 260 NOPs between its two unwind ops, pushing the IP
// offsets past 255 so EPILOG_INFO_LARGE is required. The two epilogs are
// byte-identical, so the second is encoded as an inherited (NumberOfOps == 0)
// descriptor. Per the V3 spec, Flags bits 0 and 1 are not inherited and the
// producer must replicate them, so the inherited descriptor must still carry
// the Large flag.
large_inherit:
    .seh_proc large_inherit
    .seh_unwindversion 3
    .seh_pushreg %rbx
    pushq   %rbx
    .seh_stackalloc 32
    subq    $32, %rsp
    .seh_endprologue
    callq   c
    testl   %eax, %eax
    jle     .L_LARGE_INHERIT_ELSE
    // Epilog 0: large mirror
    .seh_startepilogue
    .seh_stackalloc 32
    addq    $32, %rsp
    .rept 260
    nop
    .endr
    .seh_pushreg %rbx
    popq    %rbx
    .seh_endepilogue
    retq
.L_LARGE_INHERIT_ELSE:
    // Epilog 1: identical large mirror
    .seh_startepilogue
    .seh_stackalloc 32
    addq    $32, %rsp
    .rept 260
    nop
    .endr
    .seh_pushreg %rbx
    popq    %rbx
    .seh_endepilogue
    retq
    .seh_endproc
// CHECK-LABEL:  StartAddress: large_inherit
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog: 0x5
// CHECK-NEXT:     PayloadWords: 9
// CHECK-NEXT:     NumberOfOps: 2
// CHECK-NEXT:     NumberOfEpilogs: 2
// CHECK-NEXT:     Prolog [2 ops]:
// CHECK-NEXT:       [0] IP +0x0001: ALLOC_SMALL Size=0x20
// CHECK-NEXT:       [1] IP +0x0000: PUSH Reg=RBX
// CHECK-NEXT:     Epilog [0] {
// CHECK-NEXT:       Flags [ (0x2)
// CHECK-NEXT:         Large (0x2)
// CHECK-NEXT:       ]
// CHECK-NEXT:       EpilogOffset: -0x10A
// CHECK-NEXT:       NumberOfOps: 2
// CHECK-NEXT:       FirstOp: 0x0
// CHECK-NEXT:       IpOffsetOfLastInstruction: 0x109
// CHECK-NEXT:       [0] IP +0x0000: ALLOC_SMALL Size=0x20
// CHECK-NEXT:       [1] IP +0x0108: PUSH Reg=RBX
// CHECK-NEXT:     }
// Epilog 1 inherits, but must still carry the Large flag in its own byte.
// CHECK-NEXT:     Epilog [1] {
// CHECK-NEXT:       Flags [ (0x2)
// CHECK-NEXT:         Large (0x2)
// CHECK-NEXT:       ]
// CHECK-NEXT:       EpilogOffset: -0x214
// CHECK-NEXT:       NumberOfOps: 0
// CHECK-NEXT:       (inherits from base epilog: FirstOp=0x0, IpOffsetOfLastInstruction=0x109, 2 ops)
// CHECK-NEXT:     }
