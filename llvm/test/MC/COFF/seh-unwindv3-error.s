// RUN: not llvm-mc -triple x86_64-pc-win32 -filetype=obj %s -o /dev/null 2>&1 | FileCheck %s

// Test: invalid unwind version (not 2 or 3) inside a function.
.text
bad_version:
    .seh_proc bad_version
    .seh_unwindversion 4
// CHECK: error: Unsupported version specified in .seh_unwindversion
    .seh_endprologue
    retq
    .seh_endproc

// Test: .seh_push2regs in a V1 function (no .seh_unwindversion) produces error.
push2_in_v1:
    .seh_proc push2_in_v1
    .seh_push2regs %r12, %r13
// CHECK: error: .seh_push2regs is only supported for unwind v3
    .seh_endprologue
    retq
    .seh_endproc

// Test: .seh_push2regs missing comma between registers.
push2_missing_comma:
    .seh_proc push2_missing_comma
    .seh_unwindversion 3
    .seh_push2regs %r12 %r13
// CHECK: error: expected comma between registers
    .seh_endprologue
    retq
    .seh_endproc

// Test: .seh_push2regs missing second register.
push2_missing_reg2:
    .seh_proc push2_missing_reg2
    .seh_unwindversion 3
    .seh_push2regs %r12,
// CHECK: error: invalid register name
    .seh_endprologue
    retq
    .seh_endproc

// Test: .seh_push2regs with trailing junk.
push2_trailing_junk:
    .seh_proc push2_trailing_junk
    .seh_unwindversion 3
    .seh_push2regs %r12, %r13 extra
// CHECK: error: expected end of directive
    .seh_endprologue
    retq
    .seh_endproc

// Test: UOP_Push2 recorded under V3 then frame downgraded to V2 — the
// directive-level check passes (since the frame is already V3), so the
// error must come from the unwind-info emitter as a recoverable diagnostic
// rather than a fatal crash.
.seh_unwindversion 3
push2_downgrade_v2:
    .seh_proc push2_downgrade_v2
    .seh_push2regs %r12, %r13
    push2   %r13, %r12
    .seh_stackalloc 32
    subq    $32, %rsp
    .seh_endprologue
    // Downgrade this frame to V2; UOP_Push2 cannot be encoded for V1/V2.
    .seh_unwindversion 2
    nop
    .seh_startepilogue
    .seh_unwindv2start
    .seh_endepilogue
    retq
    .seh_endproc
// CHECK: error: UOP_Push2 (PUSH2 with two registers) requires V3 unwind info
