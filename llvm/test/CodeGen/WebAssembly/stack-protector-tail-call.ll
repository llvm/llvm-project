; RUN: llc -verify-machineinstrs -mtriple=wasm32-unknown-unknown -mattr=+tail-call < %s | FileCheck --check-prefix=TAIL -DPTR=i32 %s
; RUN: llc -verify-machineinstrs -mtriple=wasm64-unknown-unknown -mattr=+tail-call < %s | FileCheck --check-prefix=TAIL -DPTR=i64 %s

; Stack protection must check the guard before a tail transfer and keep both
; halves of the selected call together when splitting the block.
declare i32 @tail_callee(i32)

define i32 @protected_tail_direct(i32 %x) sspreq {
; TAIL-LABEL: protected_tail_direct:
; TAIL:         [[PTR]].load __stack_chk_guard
; TAIL:         [[PTR]].store
; TAIL:         [[PTR]].load __stack_chk_guard
; TAIL:         [[PTR]].load
; TAIL:         [[PTR]].eq
; TAIL:         br_if
; TAIL:         call __stack_chk_fail
; TAIL:         unreachable
; TAIL:         end_block
; TAIL:         global.set __stack_pointer
; TAIL:         return_call tail_callee
  %r = tail call i32 @tail_callee(i32 %x)
  ret i32 %r
}

define i32 @protected_tail_indirect(i32 %x, ptr %callee) sspreq {
; TAIL-LABEL: protected_tail_indirect:
; TAIL:         [[PTR]].load __stack_chk_guard
; TAIL:         [[PTR]].store
; TAIL:         [[PTR]].load __stack_chk_guard
; TAIL:         [[PTR]].load
; TAIL:         [[PTR]].eq
; TAIL:         br_if
; TAIL:         call __stack_chk_fail
; TAIL:         unreachable
; TAIL:         end_block
; TAIL:         global.set __stack_pointer
; TAIL:         return_call_indirect
  %r = tail call i32 %callee(i32 %x)
  ret i32 %r
}

; Ordinary calls must still execute before the epilogue guard check.
define i32 @protected_notail(i32 %x) sspreq {
; TAIL-LABEL: protected_notail:
; TAIL:         [[PTR]].load __stack_chk_guard
; TAIL:         [[PTR]].store
; TAIL:         {{^[[:space:]]*}}call tail_callee
; TAIL:         [[PTR]].load __stack_chk_guard
; TAIL:         [[PTR]].load
; TAIL:         [[PTR]].eq
; TAIL:         br_if
; TAIL:         call __stack_chk_fail
; TAIL:         unreachable
; TAIL:         end_block
; TAIL:         global.set __stack_pointer
; TAIL-NOT:     return_call
  %r = notail call i32 @tail_callee(i32 %x)
  ret i32 %r
}
