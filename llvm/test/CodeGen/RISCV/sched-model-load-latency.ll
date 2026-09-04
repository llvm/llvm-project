; Test that -riscv-sched-load-latency overrides the scheduler model's LoadLatency
; field, and that the override affects code generation decisions that depend on
; LoadLatency (specifically RISC-V's getMaxBuildIntsCost, which uses
; LoadLatency to decide how many instructions can be used to materialize a
; large integer constant inline vs. using the constant pool).
;
; rocket-rv64 has LoadLatency=3, so getMaxBuildIntsCost() returns 4 by
; default. The constant 0x800000007bbbbbbb requires exactly 4 MatInt
; instructions, so it is materialized inline under the default model.
;
; With -riscv-sched-load-latency=2, RISCVSubtarget::getLoadLatency() returns 2, making
; getMaxBuildIntsCost() return 3. Since 4 > 3, the constant is too expensive
; to build inline and falls back to the constant pool.
;
; RUN: llc -mtriple=riscv64 -mcpu=rocket-rv64 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefix=DEFAULT
; RUN: llc -mtriple=riscv64 -mcpu=rocket-rv64 -riscv-sched-load-latency=2 \
; RUN:   -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefix=LOWLATENCY

; With default LoadLatency=3 (maxCost=4), the 4-instruction constant is built
; inline using the MatInt sequence.
; DEFAULT-LABEL: large_int:
; DEFAULT:       # %bb.0:
; DEFAULT-NEXT:    lui a0, 506812
; DEFAULT-NEXT:    addi a0, a0, -1093
; DEFAULT-NEXT:    slli a1, a0, 63
; DEFAULT-NEXT:    add a0, a0, a1
; DEFAULT-NEXT:    ret

; With -riscv-sched-load-latency=2 (maxCost=3), the 4-instruction sequence exceeds
; the cost threshold, so the constant is loaded from the constant pool.
; LOWLATENCY-LABEL: large_int:
; LOWLATENCY:       # %bb.0:
; LOWLATENCY-NEXT:    lui a0, %hi(.LCPI0_0)
; LOWLATENCY-NEXT:    ld a0, %lo(.LCPI0_0)(a0)
; LOWLATENCY-NEXT:    ret

define i64 @large_int() {
  ; 0x800000007bbbbbbb = -9223372034778874949
  ; Requires exactly 4 MatInt instructions: lui, addi, slli, add
  ret i64 -9223372034778874949
}
