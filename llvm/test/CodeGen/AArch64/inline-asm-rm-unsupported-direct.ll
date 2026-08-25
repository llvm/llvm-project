; RUN: not llc -mtriple=aarch64-unknown-linux-gnu --global-isel=true --fast-isel=false --regalloc=greedy < %s 2>&1 \
; RUN:     | FileCheck %s
; RUN: not llc -mtriple=aarch64-unknown-linux-gnu --global-isel=false --fast-isel=true --regalloc=greedy < %s 2>&1 \
; RUN:     | FileCheck %s
; RUN: not llc -mtriple=aarch64-unknown-linux-gnu --global-isel=false --fast-isel=false --regalloc=greedy < %s 2>&1 \
; RUN:     | FileCheck %s
; RUN: not llc -mtriple=aarch64-unknown-linux-gnu --global-isel=false --fast-isel=false --regalloc=fast < %s 2>&1 \
; RUN:     | FileCheck %s

; A *direct* (non-indirect) "=rm" output -- something Clang itself would
; never emit for this target, precisely because
; TargetLowering::supportsRegMemInlineAsmFolding() is false for it and
; CGStmt.cpp mirrors that check -- is still directly constructible in IR, by
; another frontend or by hand, as here. Regression guard for a case that
; used to hard-crash instead of failing cleanly: an UNREACHABLE in
; SelectionDAGBuilder's computeConstraintToUse() for the SelectionDAG
; frameworks, and a null CallOperandVal dereference in
; InlineAsmLowering::lowerInlineAsm() for GlobalISel. A clean diagnostic is
; the correct behavior for a target with no register-pressure fallback to
; offer.

; CHECK: error: unsupported inline asm: constraint 'm' cannot be satisfied in a register and has no memory to fall back to
define i64 @test_rm_output_direct_unsupported() {
entry:
  %0 = call i64 asm "// $0", "=rm"()
  ret i64 %0
}
