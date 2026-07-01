; RUN: llc -mtriple=aarch64-linux-gnu -verify-machineinstrs < %s | FileCheck %s

; The preserve_all calling convention promotes almost all registers to
; callee-saved, so a function that clobbers them all spills a callee-save area
; larger than the +/-504 byte reach of the paired LDP/STP scaled 7-bit
; immediate. Registers whose offset exceeds that range must be spilled and
; reloaded with a single STR/LDR (which has a wider 12-bit immediate) instead
; of an out-of-range STP/LDP.
;
; This also covers the related callee-save size accounting: D8-D15 (AAPCS) and
; the enclosing Q8-Q15 (preserve_all) must not be double-counted, otherwise the
; frame is over-sized and the prologue emits a redundant SP adjustment.

; CHECK-LABEL: trigger_stack_spill:
; The last in-range pair sits at the +504 boundary; the registers above it are
; spilled as single STR (offset > 504, unencodable as a pair).
; CHECK:      stp x22, x21, [sp, #504]
; CHECK-NEXT: str x20, [sp, #520]
; CHECK-NEXT: str x19, [sp, #528]
; The unwind offsets must match where the registers are actually stored.
; CHECK:      .cfi_offset w19, -16
; CHECK:      .cfi_offset w20, -24
; The epilogue reloads them symmetrically with single LDR.
; CHECK-DAG:  ldr x19, [sp, #528]
; CHECK-DAG:  ldr x20, [sp, #520]

define preserve_allcc void @trigger_stack_spill() {
entry:
  call void asm sideeffect "", "~{x0},~{x1},~{x2},~{x3},~{x4},~{x5},~{x6},~{x7},~{x8},~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x15},~{x16},~{x17},~{x18},~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28},~{fp},~{lr},~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31},~{memory},~{cc}"()
  ret void
}
