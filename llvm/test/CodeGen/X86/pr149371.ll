; RUN: not llc -mtriple=x86_64 < %s 2>&1 | FileCheck %s

; Inline asm whose x87 register constraints don't describe a valid stack
; layout must be diagnosed without crashing the FP stackifier afterwards.

; CHECK: error: {{.*}}: fixed input regs must be last on the x87 stack
define void @fixed_input(double %y) {
  call void asm sideeffect "", "{st(1)},~{dirflag},~{fpsr},~{flags}"(double %y)
  ret void
}

; CHECK: error: {{.*}}: output regs must be last on the x87 stack
define double @output_regs() {
  %r = call double asm sideeffect "", "={st(1)},~{dirflag},~{fpsr},~{flags}"()
  ret double %r
}

; CHECK: error: {{.*}}: clobbers must be last on the x87 stack
define double @clobbers() {
  %r = call double asm sideeffect "", "={st},~{st(2)},~{dirflag},~{fpsr},~{flags}"()
  ret double %r
}

; PR149371: the only x87 input is tied to st(1) while st(0) and st(1) are
; both outputs. Simulating the pops and pushes for this layout used to leave
; a stale entry in the stack model, and the second asm then tripped
; "Live count mismatch" at the return.
; CHECK: error: {{.*}}: fixed input regs must be last on the x87 stack
; CHECK-NEXT: error: {{.*}}: implicitly popped regs must be last on the x87 stack
; CHECK-NEXT: error: {{.*}}: fixed input regs must be last on the x87 stack
; CHECK-NEXT: error: {{.*}}: implicitly popped regs must be last on the x87 stack
; CHECK-NOT: Assertion
; CHECK-NOT: LLVM ERROR
define void @popped(double %x, double %y) {
  %r1 = call { double, double } asm sideeffect "", "={st},={st(1)},R0,1,~{dirflag},~{fpsr},~{flags}"(double %x, double %y)
  %r2 = call { double, double } asm sideeffect "", "={st},={st(1)},R0,1,~{dirflag},~{fpsr},~{flags}"(double %x, double %y)
  ret void
}
