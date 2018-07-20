; RUN: not llc -march=sparc <%s 2>&1 | FileCheck %s
; RUN: not llc -march=sparcv9 <%s 2>&1 | FileCheck %s

; CHECK: error: couldn't allocate input reg for constraint '{f32}'
; CHECK: error: couldn't allocate input reg for constraint '{f21}'
; CHECK: error: couldn't allocate input reg for constraint '{f38}'
define void @test_constraint_float_reg() {
entry:
  tail call void asm sideeffect "fadds $0,$1,$2", "{f32},{f0},{f0}"(float 6.0, float 7.0, float 8.0)
  tail call void asm sideeffect "faddd $0,$1,$2", "{f21},{f0},{f0}"(double 9.0, double 10.0, double 11.0)
  tail call void asm sideeffect "faddq $0,$1,$2", "{f38},{f0},{f0}"(fp128 0xL0, fp128 0xL0, fp128 0xL0)
  ret void
}
