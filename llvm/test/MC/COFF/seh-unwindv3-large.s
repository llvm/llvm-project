// RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj %s | llvm-readobj -u - | FileCheck %s

// Tests for V3 UNW_FLAG_LARGE emission when prolog exceeds 255 bytes.
// This exercises the LARGE prolog header (5-byte) and 16-bit IP offsets.

// CHECK:       UnwindInformation [

.text

// --- Test 1: prolog with IP offset > 255 (evaluatable case) ---
// Uses a large .space to push the second prolog instruction past offset 255.
// This should trigger UNW_FLAG_LARGE with known values.
large_prolog_known:
    .seh_proc large_prolog_known
    .seh_unwindversion 3
    .seh_pushreg %rbx
    pushq   %rbx
    // Pad with 260 bytes of NOPs to push the next IP offset past 255.
    .rept 260
    nop
    .endr
    .seh_stackalloc 32
    subq    $32, %rsp
    .seh_endprologue
    nop
    .seh_startepilogue
    .seh_stackalloc 32
    addq    $32, %rsp
    .seh_pushreg %rbx
    popq    %rbx
    .seh_endepilogue
    retq
    .seh_endproc
// CHECK-LABEL:  StartAddress: large_prolog_known
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x8)
// CHECK-NEXT:       Large (0x8)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog: 0x109
// CHECK-NEXT:     PayloadWords: 8
// CHECK-NEXT:     NumberOfOps: 2
// CHECK-NEXT:     NumberOfEpilogs: 1
// CHECK:          Prolog [2 ops]:
// CHECK-NEXT:       [0] IP +0x0105: ALLOC_SMALL Size=0x20
// CHECK-NEXT:       [1] IP +0x0000: PUSH Reg=RBX
// CHECK:          Epilog [0] {
// CHECK:            EpilogOffset: -0x6
// CHECK-NEXT:       NumberOfOps: 2
// CHECK-NEXT:       FirstOp: 0x0
// CHECK-NEXT:       IpOffsetOfLastInstruction: 0x5
// CHECK-NEXT:       [0] IP +0x0000: ALLOC_SMALL Size=0x20
// CHECK-NEXT:       [1] IP +0x0004: PUSH Reg=RBX
// Uses a forward jump inside the prolog to create a relaxable fragment that
// makes GetOptionalAbsDifference return nullopt, triggering the conservative
// NeedsLargeProlog=true path with fixup-based emission. (Alignment
// directives no longer work for this: their padding is resolved exactly
// before layout nowadays.)
large_prolog_unevaluatable:
    .seh_proc large_prolog_unevaluatable
    .seh_unwindversion 3
    .seh_pushreg %rbx
    pushq   %rbx
    // A not-yet-relaxed forward jump ΓÇö the distance from the function start
    // to subsequent instructions is unevaluatable when unwind info is
    // emitted at .seh_endproc time.
    jmp 1f
    .rept 252
    nop
    .endr
1:
    .seh_stackalloc 32
    subq    $32, %rsp
    .seh_endprologue
    nop
    .seh_startepilogue
    .seh_stackalloc 32
    addq    $32, %rsp
    .seh_pushreg %rbx
    popq    %rbx
    .seh_endepilogue
    retq
    .seh_endproc
// When the expression is unevaluatable, we conservatively set LARGE.
// The output should still be valid V3 unwind info with the Large flag.
// After layout the jmp relaxes to a 5-byte near jump: the pushq is at
// func+0, and the subq lands at func+0x102 (1 + 5 + 252), so the fixups
// resolve to values that genuinely need the 16-bit LARGE encodings.
// CHECK-LABEL:  StartAddress: large_prolog_unevaluatable
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x8)
// CHECK-NEXT:       Large (0x8)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog: 0x106
// CHECK-NEXT:     PayloadWords: 8
// CHECK-NEXT:     NumberOfOps: 2
// CHECK-NEXT:     NumberOfEpilogs: 1
// CHECK:          Prolog [2 ops]:
// CHECK-NEXT:       [0] IP +0x0102: ALLOC_SMALL Size=0x20
// CHECK-NEXT:       [1] IP +0x0000: PUSH Reg=RBX
// CHECK:          Epilog [0] {
// CHECK:            EpilogOffset: -0x6
// CHECK-NEXT:       NumberOfOps: 2
// CHECK-NEXT:       FirstOp: 0x0
// CHECK-NEXT:       IpOffsetOfLastInstruction: 0x5
// CHECK-NEXT:       [0] IP +0x0000: ALLOC_SMALL Size=0x20
// CHECK-NEXT:       [1] IP +0x0004: PUSH Reg=RBX

// --- Test 3: epilog with IP offset > 255 (evaluatable case) ---
// The epilog has 260 NOPs between the two unwind operations, pushing the
// second epilog IP offset and IpOffsetOfLastInstruction past 255.
// This should trigger EPILOG_INFO_LARGE (bit 1 in epilog flags) with 16-bit
// epilog IP offsets and IpOffsetOfLastInstruction.
// The prolog is small, so UNW_FLAG_LARGE should NOT be set.
large_epilog_known:
    .seh_proc large_epilog_known
    .seh_unwindversion 3
    .seh_pushreg %rbx
    pushq   %rbx
    .seh_stackalloc 32
    subq    $32, %rsp
    .seh_endprologue
    nop
    .seh_startepilogue
    .seh_stackalloc 32
    addq    $32, %rsp
    // Pad with 260 NOPs inside the epilog to push the popq past offset 255.
    .rept 260
    nop
    .endr
    .seh_pushreg %rbx
    popq    %rbx
    .seh_endepilogue
    retq
    .seh_endproc
// CHECK-LABEL:  StartAddress: large_epilog_known
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog: 0x5
// CHECK-NEXT:     PayloadWords: 8
// CHECK-NEXT:     NumberOfOps: 2
// CHECK-NEXT:     NumberOfEpilogs: 1
// CHECK:          Prolog [2 ops]:
// CHECK-NEXT:       [0] IP +0x0001: ALLOC_SMALL Size=0x20
// CHECK-NEXT:       [1] IP +0x0000: PUSH Reg=RBX
// CHECK:          Epilog [0] {
// CHECK-NEXT:       Flags [ (0x2)
// CHECK-NEXT:         Large (0x2)
// CHECK-NEXT:       ]
// CHECK-NEXT:       EpilogOffset: -0x10A
// CHECK-NEXT:       NumberOfOps: 2
// CHECK-NEXT:       FirstOp: 0x0
// CHECK-NEXT:       IpOffsetOfLastInstruction: 0x109
// CHECK-NEXT:       [0] IP +0x0000: ALLOC_SMALL Size=0x20
// CHECK-NEXT:       [1] IP +0x0108: PUSH Reg=RBX

// --- Test 4: epilog with a relaxable fragment (unevaluatable case) ---
// Uses a forward jump inside the epilog to create a relaxable fragment,
// making the epilog IP offsets unevaluatable and triggering
// EPILOG_INFO_LARGE conservatively.
large_epilog_unevaluatable:
    .seh_proc large_epilog_unevaluatable
    .seh_unwindversion 3
    .seh_pushreg %rbx
    pushq   %rbx
    .seh_stackalloc 32
    subq    $32, %rsp
    .seh_endprologue
    nop
    .seh_startepilogue
    .seh_stackalloc 32
    addq    $32, %rsp
    // A not-yet-relaxed forward jump inside the epilog makes offsets
    // unevaluatable.
    jmp 2f
    .rept 252
    nop
    .endr
2:
    .seh_pushreg %rbx
    popq    %rbx
    .seh_endepilogue
    retq
    .seh_endproc
// CHECK-LABEL:  StartAddress: large_epilog_unevaluatable
// CHECK:        UnwindInfo {
// CHECK-NEXT:     Version: 3
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     SizeOfProlog: 0x5
// CHECK-NEXT:     PayloadWords: 8
// CHECK-NEXT:     NumberOfOps: 2
// CHECK-NEXT:     NumberOfEpilogs: 1
// CHECK:          Prolog [2 ops]:
// CHECK-NEXT:       [0] IP +0x0001: ALLOC_SMALL Size=0x20
// CHECK-NEXT:       [1] IP +0x0000: PUSH Reg=RBX
// CHECK:          Epilog [0] {
// CHECK-NEXT:       Flags [ (0x2)
// CHECK-NEXT:         Large (0x2)
// CHECK-NEXT:       ]
// CHECK-NEXT:       EpilogOffset: -0x107
// CHECK-NEXT:       NumberOfOps: 2
// CHECK-NEXT:       FirstOp: 0x0
// CHECK-NEXT:       IpOffsetOfLastInstruction: 0x106
// CHECK-NEXT:       [0] IP +0x0000: ALLOC_SMALL Size=0x20
// CHECK-NEXT:       [1] IP +0x0105: PUSH Reg=RBX
