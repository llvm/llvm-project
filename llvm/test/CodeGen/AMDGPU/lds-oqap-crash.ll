; RUN: llc < %s -march=r600 -mcpu=redwood -verify-machineinstrs | FileCheck %s

; The test is for a bug in R600EmitClauseMarkers.cpp where this pass
; was searching for a use of the OQAP register in order to determine
; if an LDS instruction could fit in the current clause, but never finding
; one.  This created an infinite loop and hung the compiler.
;
; The LDS instruction should not have been defining OQAP in the first place,
; because the LDS instructions are pseudo instructions and the OQAP
; reads and writes are bundled together in the same instruction.

; CHECK: {{^}}lds_crash:
define amdgpu_kernel void @lds_crash(i32 addrspace(1)* %out, i32 addrspace(3)* %in, i32 %a, i32 %b, i32 %c) {
entry:
  %0 = load i32, i32 addrspace(3)* %in
  ; This block needs to be > 115 ISA instructions to hit the bug,
  ; so we'll use udiv instructions.
  %div0 = udiv i32 %0, %b
  %div1 = udiv i32 %div0, %a
  %div2 = udiv i32 %div1, 11
  %div3 = udiv i32 %div2, %a
  %div4 = udiv i32 %div3, %b
  %div5 = udiv i32 %div4, %c
  %div6 = udiv i32 %div5, %div0
  %div7 = udiv i32 %div6, %div1
  store i32 %div7, i32 addrspace(1)* %out
  ret void
}
