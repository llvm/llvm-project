; RUN: not llc -mtriple=i686-unknown-linux-gnu -regalloc=pbqp \
; RUN:   -verify-machineinstrs < %s 2>&1 | FileCheck %s

; The asm's operands need more registers than are available, so no solution
; of the PBQP problem has a finite cost. That's an error, not an assertion in
; the solver. The error names the register that no solution could allocate,
; which here is one that a spill created.

; CHECK: error: {{.*}}ran out of registers during register allocation

define { i32, i32 } @f(i32 %a1, i8 %0, ptr %v3) {
  %v1 = mul i32 %a1, 394
  %r = call { i32, i32 } asm sideeffect "", "=r,=r,{ebx},m,m,r,{edi},0,1,~{esi},~{eax},~{ebp},~{edx}"(i32 0, ptr %v3, i8 %0, ptr null, i32 %a1, i32 %v1, i32 0)
  ret { i32, i32 } %r
}
