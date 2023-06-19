; RUN: llc -march=hexagon -filetype=obj < %s | llvm-objdump -d - | FileCheck %s

; The output assembly (textual) contains the instruction
;   r29 = add(r29,#4294967136)
; The value 4294967136 is -160 when interpreted as a signed 32-bit
; integer, so it fits in the range of the immediate operand without
; a constant extender. The range check in HexagonInstrInfo was putting
; the operand value into an int variable, reporting no need for an
; extender. This resulted in a packet with 4 instructions, including
; the "add". The corresponding check in HexagonMCInstrInfo was using
; an int64_t variable, causing an extender to be emitted when lowering
; to MCInst, and resulting in a packet with 5 instructions.

; Check that this doesn't crash.
; CHECK: r29 = add(r29,#-0xa0)

target triple = "hexagon-unknown-linux-gnu"

define float @f0() {
b0:
  %v0 = alloca i8, i32 0, align 1
  %v1 = alloca float, i32 -42, align 4
  %v2 = load float, ptr %v1, align 4
  store i8 0, ptr %v0, align 1
  ret float %v2
}
